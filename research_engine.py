"""Integrated, resumable multi-asset research batch; no live orders or holdout access.
Historical experiment scripts remain intact. One batch report per protocol.
"""
import os
os.environ.setdefault("POLARS_MAX_THREADS","1")
import json, zipfile, traceback, hashlib, time, inspect, gc, io
import psutil
from pathlib import Path
import numpy as np
import polars as pl
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from baseline_engine import digest
from feature_engine import technical, BASE_FEATURES
from validation_protocol import Fold, make_folds
from experiment_store import identity, checkpoint
from robustness_engine import A,B,DAY,ledger,ci
from funding_repair_engine import apply
from data_engine import aggregate
from execution_engine import score_trades
from resource_policy import budget,tree_rss
from universe import CRYPTO_SYMBOLS

ROOT=Path("data")
MIN=60000
H=60*MIN
ASSETS=["BTCUSDT","ETHUSDT","BNBUSDT","TRXUSDT","NEARUSDT"]
TFS={"30m":30*MIN,"1h":H,"4h":4*H}
HORIZONS=(1,4,8,24)
THRESHOLDS=(.55,.60,.65,.70)
RULES=("fixed","half_invalidation")
COLS=[c for c in BASE_FEATURES if c!="body"]
COSTS=((5,1),(10,5))
SPEC={"timeframes":TFS,"horizons_hours":HORIZONS,"thresholds":THRESHOLDS,
      "models":["logistic","shallow_tree"],"targets":["binary","cost_aware"],"coarse_C":1.,"refine_C":[.1,10.],
      "tree":{"max_depth":3,"min_samples_leaf":.05},"rule_features":["ret_4","ret_16"],"rule_quantiles":[.80,.95],"rule_directions":["momentum","reversal"],
      "refine_families":1,"exit_rules":RULES,"features":COLS,
      "delay_minutes":15,"train_days":180,"inner_fit_days":120,"inner_validation_days":30,"inner_windows":2,
      "outer_test_days":30,"seed":42,"edge_shrinkage_zero_prior_trades":10,
      "objective":"positive primary net nonoverlap expectancy; rank primary expectancy shrunk toward zero for sparse evidence across two chronological inner windows; stress sensitivity reported, not a hard veto; abstain if none; no frequency quota",
      "costs_bps":COSTS,
      "execution":"decision then 15m delay; retain all signal opportunities; separate per-symbol decision-time nonoverlap diagnostic; no shared capital, leverage, margin, or slot constraints",
      "funding":"strict 8h schedule; no unknown funding imputation; common known-cost inner validation mask across horizons/exit variants; outer retains all signals, unknown outcomes explicitly conditional",
      "holdout_excluded":True,"inner_comparison":"common maximum horizon end within each validation window",
      "family_comparison":"each family selected independently on inner data and evaluated on outer data; joint winner chosen before outer"}

def require(ok,msg):
    if not ok:raise ValueError(msg)

def read_report(name):
    with zipfile.ZipFile(ROOT/"reports"/name) as z:return json.loads(z.read("metrics.json"))

def monthly(directory):
    return [p for p in sorted(directory.glob("*.parquet")) if "2021-01"<=p.stem<="2024-12"]

def load_funding(symbol):
    fp=monthly(ROOT/"funding"/symbol)
    require(len(fp)==48,"Missing funding partitions "+symbol)
    if symbol=="BTCUSDT":
        f=apply(ledger(ROOT)).sort("event_time")
    else:
        parts=[]
        for p in fp:
            x=pl.read_parquet(p).select("event_time","funding_rate","funding_interval_hours")
            m=pl.read_parquet(ROOT/"mark"/symbol/p.name,columns=["event_time","open"]).rename({"event_time":"minute","open":"mark_price"})
            parts.append(x.with_columns((pl.col("event_time")//MIN*MIN).alias("minute")).join(m,on="minute",how="left",validate="m:1"))
        f=pl.concat(parts).filter(pl.col("event_time").is_between(A,B,closed="left")).sort("event_time")
        require((f["funding_interval_hours"]==8).fill_null(False).all(),"Nonstandard funding schedule "+symbol)
        f=f.with_columns(pl.lit(symbol).alias("symbol")).select("symbol","event_time","funding_rate","mark_price")
    ts=f["event_time"].to_numpy()
    expected=np.arange(A,B,8*H)
    require(len(ts)==len(expected) and np.all(np.abs(ts-expected)<=MIN) and len(np.unique(ts))==len(ts),"Incomplete funding schedule "+symbol)
    rate=f["funding_rate"].to_numpy()
    require(np.isfinite(rate).all(),"Invalid funding rate "+symbol)
    price=f["mark_price"].cast(pl.Float64).to_numpy()
    good=np.isfinite(price)&(price>0)
    return f,ts,np.r_[0.,np.cumsum(np.where(good,rate*price,0.))],np.r_[0,np.cumsum(~good)]

def costs(entry,end,ep,xp,side,funding,fee,slip):
    _,ts,cum,bad=funding
    a=np.searchsorted(ts,entry,side="right");b=np.searchsorted(ts,end,side="right")
    u=ep*(1+side*slip/10000);v=xp*(1-side*slip/10000)
    r=side*(v-u)/u-fee/10000*(u+v)/u-side*(cum[b]-cum[a])/u
    return np.where(bad[b]==bad[a],r,np.nan)

def base_arrays(symbol,raw,cache):
    """Immutable source-keyed feature/price cache, reused on restart."""
    cache.mkdir(parents=True,exist_ok=True)
    path=cache/"base.npz";meta=cache/"base.json"
    if path.exists() and meta.exists() and json.loads(meta.read_text()).get("sha256")==digest([path]):
        with np.load(path,allow_pickle=False) as z:
            return {k:z[k] for k in z.files}
    bars={tf:[] for tf in TFS};prices=[];cursor=A
    for p in raw:
        f=pl.read_parquet(p).sort("open_time")
        t=f["open_time"].to_numpy()
        require(len(t)>0 and t[0]==cursor and np.all(np.diff(t)==MIN) and np.all(t%MIN==0),"Raw gap/duplicate "+symbol+" "+p.stem)
        cursor=int(t[-1])+MIN
        v=f["open"].cast(pl.Float64).to_numpy()
        require(np.isfinite(v).all() and (v>0).all(),"Invalid raw opens")
        prices.append(v)
        for tf in TFS:bars[tf].append(aggregate(f,tf))
    require(cursor==B,"Incomplete raw development grid "+symbol)
    arrays={"prices":np.concatenate(prices)}
    for tf,step in TFS.items():
        h=technical(pl.concat(bars[tf]).sort("open_time"),step)
        h=h.filter(pl.all_horizontal([pl.col(c).is_finite().fill_null(False) for c in BASE_FEATURES]))
        h=h.filter(pl.col("decision_time")+15*MIN+max(HORIZONS)*H<B)
        arrays[tf+"_d"]=h["decision_time"].to_numpy()
        arrays[tf+"_x"]=h.select(COLS).to_numpy()
    tmp=path.with_suffix(".tmp")
    with tmp.open("wb") as f:np.savez(f,**arrays)
    tmp.replace(path)
    from data_engine import write_json
    write_json(meta,{"sha256":digest([path])})
    return arrays

def build(symbol,raw,funding,cache):
    arrays=base_arrays(symbol,raw,cache);prices=arrays["prices"]
    require(len(prices)==(B-A)//MIN,"Invalid cached price grid")
    def px(t):
        require(np.all(t>=A)&np.all(t<B)&np.all((t-A)%MIN==0),"Execution outside raw grid")
        return prices[(t-A)//MIN]
    datasets={}
    for tf,step in TFS.items():
        d=arrays[tf+"_d"];e=d+15*MIN;x=arrays[tf+"_x"]
        require(len(d)==len(x) and np.isfinite(x).all() and np.all(np.diff(d)>0),"Invalid cached features")
        ep=px(e)
        for horizon in HORIZONS:
            z=e+horizon*H;xp=px(z);mid=e+horizon*H//2
            variants={}
            for side in (-1,1):
                q=np.where(side*(px(mid)/ep-1)<=0,mid+MIN,z)
                for rule,end in [("fixed",z),("half_invalidation",q)]:
                    exit_px=px(end)
                    variants[(side,rule)]=(end,exit_px,*[costs(e,end,ep,exit_px,side,funding,fee,slip) for fee,slip in COSTS])
            l=variants[(1,"fixed")][2];s=variants[(-1,"fixed")][2]
            require(not np.any((l>0)&(s>0)),"Ambiguous cost target")
            y=np.where(l>0,1,np.where(s>0,-1,0))
            # Funding availability is an ex-post evaluation mask, never an outer entry feature.
            _,ft,_,bad=funding
            a=np.searchsorted(ft,e,side="right");b=np.searchsorted(ft,e+max(HORIZONS)*H,side="right")
            common_known=bad[a]==bad[b]
            datasets[(tf,horizon)]={"d":d,"e":e,"z":z,"ep":ep,"xp":xp,"x":x,"common_known":common_known,
                "binary":np.where(xp>ep,1,-1),"cost_aware":y,"known":np.isfinite(l)&np.isfinite(s),
                "variants":variants}
    return datasets

def predict(ds,target,C,train,test,model_name="logistic"):
    # Binary direction needs only price labels; cost-aware target needs known realized costs.
    if target=="cost_aware":train=train&ds["known"]
    y=ds[target][train]
    expected=np.array([-1,1] if target=="binary" else [-1,0,1])
    if train.sum()<100 or test.sum()<30 or not np.array_equal(np.unique(y),expected):
        return None
    with threadpool_limits(limits=1):
        scale=StandardScaler()
        x=scale.fit_transform(ds["x"][train]);v=scale.transform(ds["x"][test])
        if model_name=="logistic":model=LogisticRegression(C=C,max_iter=500,random_state=42)
        elif model_name=="shallow_tree":model=DecisionTreeClassifier(**SPEC["tree"],random_state=42)
        else:raise ValueError("Unknown model "+model_name)
        model.fit(x,y)
        p=model.predict_proba(v)
    require(np.array_equal(model.classes_,expected) and np.isfinite(p).all(),"Invalid predictions")
    return p[:,0],p[:,-1]

def opportunities(ds,test,p,threshold,rule,side=None):
    ix=np.flatnonzero(test)
    if side is None:side=np.where(p[1]>=threshold,1,np.where(p[0]>=threshold,-1,0))
    require(len(side)==len(ix),"Signal mask mismatch")
    hit=np.flatnonzero(side)
    result=[]
    for j in hit:
        i=ix[j];v=int(side[j]);end,xp,standard,stress=ds["variants"][(v,rule)]
        result.append({"decision_time":int(ds["d"][i]),"entry_time":int(ds["e"][i]),
            "label_end":int(end[i]),"entry_price":float(ds["ep"][i]),"exit_price":float(xp[i]),"side":v,
            "p_short":float(p[0][j]) if p is not None else None,"p_long":float(p[1][j]) if p is not None else None,"threshold":threshold,"exit_rule":rule,
            "opposite_standard":float(ds["variants"][(-v,rule)][2][i]) if np.isfinite(ds["variants"][(-v,rule)][2][i]) else None,
            "opposite_stress":float(ds["variants"][(-v,rule)][3][i]) if np.isfinite(ds["variants"][(-v,rule)][3][i]) else None,
            "standard":float(standard[i]) if np.isfinite(standard[i]) else None,
            "stress":float(stress[i]) if np.isfinite(stress[i]) else None})
    return result

def nonoverlap(signals):
    """Diagnostic only: independent flat-at-decision rule within each symbol."""
    free={};selected=[]
    for t in sorted(signals,key=lambda t:(t["decision_time"],t.get("symbol",""))):
        symbol=t.get("symbol","")
        if t["decision_time"]>=free.get(symbol,-1):
            selected.append(t);free[symbol]=t["label_end"]
    return selected

def summarize_inner(nsignals,primary,stress):
    n=len(primary)
    unknown=int(np.sum(~np.isfinite(primary)|~np.isfinite(stress)))
    row={"signals":int(nsignals),"nonoverlap_signals":n,"unknown_funding_trades":unknown,"score":None}
    if not n or unknown:
        row["reason"]="no_signals" if not n else "unknown_funding"
        return row
    row.update(primary_expectancy=float(primary.mean()),stress_expectancy=float(stress.mean()),
               primary_pnl=float(primary.sum()),stress_pnl=float(stress.sum()))
    if primary.mean()>0:
        row["score"]=float(primary.sum()/(n+SPEC["edge_shrinkage_zero_prior_trades"]))
    else:row["reason"]="nonpositive_primary_expectancy"
    return row

def inner_score(signals,first,last):
    selected=nonoverlap(signals)
    require(all(first<=t["decision_time"]<t["label_end"]<last for t in selected),"Inner signal outside validation")
    return summarize_inner(len(signals),np.asarray([t["standard"] for t in selected],dtype=float),
                           np.asarray([t["stress"] for t in selected],dtype=float))

def inner_outcomes(ds,test,p,threshold,rule,side=None):
    """Vectorized candidate outcomes; avoid building Python trade ledgers for every trial."""
    ix=np.flatnonzero(test)
    if side is None:side=np.where(p[1]>=threshold,1,np.where(p[0]>=threshold,-1,0))
    require(len(ix)==len(side),"Signal mask mismatch")
    hit=np.flatnonzero(side)
    i=ix[hit];v=side[hit]
    long=ds["variants"][(1,rule)];short=ds["variants"][(-1,rule)]
    ends=np.where(v==1,long[0][i],short[0][i])
    primary=np.where(v==1,long[2][i],short[2][i])
    stress=np.where(v==1,long[3][i],short[3][i])
    times=ds["d"][i]
    keep=[];j=0
    while j<len(i):
        keep.append(j)
        # label_end is strictly later than the decision, so search advances.
        j=int(np.searchsorted(times,ends[j],side="left"))
    keep=np.asarray(keep,dtype=int)
    return len(i),primary[keep],stress[keep]

def rule_signals(ds,train,test,feature,quantile,direction):
    """Causal magnitude cutoff fitted on historical features only; no fake probabilities."""
    col=COLS.index(feature);x=ds["x"][:,col]
    if train.sum()<100 or test.sum()<30:return None
    cutoff=float(np.quantile(np.abs(x[train]),quantile))
    v=x[test];side=np.where(np.abs(v)>cutoff,np.sign(v),0).astype(int)
    return side if direction=="momentum" else -side

def inner_masks(ds,fold):
    """Two forward windows, purge actual full target end before every boundary."""
    first=fold.train_start+120*DAY
    for lo in (first,first+30*DAY):
        hi=lo+30*DAY
        require(hi<=fold.test_start,"Inner window exceeds outer train")
        train=(ds["d"]>=fold.train_start)&(ds["z"]<lo)
        candidate=(ds["d"]>=lo)&(ds["e"]+max(HORIZONS)*H<hi)
        yield train,candidate&ds["common_known"],lo,hi,int(candidate.sum())

def search_fold(symbol,ds_all,fold,fit_baseline=False):
    trials=[];baseline=[];first=fold.train_start+120*DAY
    if fit_baseline:
        ds=ds_all[("1h",4)]
        train=(ds["d"]>=fold.train_start)&(ds["z"]<fold.test_start)
        test=(ds["d"]>=fold.test_start)&(ds["z"]<fold.test_end)
        p=predict(ds,"binary",1.,train,test)
        require(p is not None,"Cannot fit fixed baseline")
        baseline=opportunities(ds,test,p,.7,"fixed")
        for t in baseline:t.update(symbol=symbol,fold=fold.fold_id,tf="1h",horizon_hours=4,target="binary",C=1.,model="logistic")

    def evaluate(tf,h,model,target=None,C=None,feature=None,quantile=None,direction=None):
        ds=ds_all[(tf,h)]
        windows=[]
        spec={"tf":tf,"horizon":h,"model":model,"target":target,"C":C,
              "feature":feature,"quantile":quantile,"direction":direction}
        for train,val,lo,hi,total in inner_masks(ds,fold):
            if model=="rule":
                side=rule_signals(ds,train,val,feature,quantile,direction);p=None
                if side is None:break
            else:
                p=predict(ds,target,C,train,val,model);side=None
                if p is None:break
            windows.append((val,p,side,lo,hi,total))
        if len(windows)!=2:
            trials.append({**spec,"score":None,"reason":"insufficient rows/classes"});return
        # Threshold/exit search reuses the same trained predictions.
        for rule in RULES:
            for th in ([None] if model=="rule" else THRESHOLDS):
                count=0;prim=[];stress=[];diagnostics=[]
                for val,p,side,lo,hi,total in windows:
                    n,a,b=inner_outcomes(ds,val,p,th,rule,side)
                    count+=n;prim.append(a);stress.append(b)
                    d=summarize_inner(n,a,b)
                    diagnostics.append({"start":lo,"end":hi,"eligible_rows":int(val.sum()),
                                        "candidate_rows":total,**d})
                row=summarize_inner(count,np.concatenate(prim),np.concatenate(stress))
                trials.append({**spec,"exit":rule,"threshold":th,"inner_windows":diagnostics,**row})
    for tf,h in ds_all:
        for model in SPEC["models"]:
            for target in SPEC["targets"]:
                evaluate(tf,h,model,target,1. if model=="logistic" else None)
        for feature in SPEC["rule_features"]:
            for quantile in SPEC["rule_quantiles"]:
                for direction in SPEC["rule_directions"]:
                    evaluate(tf,h,"rule",feature=feature,quantile=quantile,direction=direction)
    # Refine the best eligible logistic family even when a rule/tree wins coarse search.
    eligible=[r for r in trials if r["score"] is not None and r["model"]=="logistic"]
    if eligible:
        best=max(eligible,key=lambda r:r["score"])
        for C in SPEC["refine_C"]:evaluate(best["tf"],best["horizon"],"logistic",best["target"],C)
    eligible=[r for r in trials if r["score"] is not None]
    family_choices={}
    family_signals={}
    def outer(best):
        ds=ds_all[(best["tf"],best["horizon"])]
        train=(ds["d"]>=fold.train_start)&(ds["z"]<fold.test_start)
        test=(ds["d"]>=fold.test_start)&(ds["z"]<fold.test_end)
        if best["model"]=="rule":
            side=rule_signals(ds,train,test,best["feature"],best["quantile"],best["direction"]);p=None
            require(side is not None,"Selected rule cannot refit")
        else:
            p=predict(ds,best["target"],best["C"],train,test,best["model"]);side=None
            require(p is not None,"Selected model cannot refit")
        signals=opportunities(ds,test,p,best["threshold"],best["exit"],side)
        for t in signals:
            t.update(symbol=symbol,fold=fold.fold_id,tf=best["tf"],horizon_hours=best["horizon"],
                     target=best["target"],C=best["C"],model=best["model"],feature=best["feature"],
                     quantile=best["quantile"],direction=best["direction"],inner_edge_score=best["score"])
        return signals
    best=max(eligible,key=lambda r:r["score"]) if eligible else None
    # All choices fixed before looking at outer outcomes; reuse winner's predictions.
    for model in (*SPEC["models"],"rule"):
        candidates=[r for r in eligible if r["model"]==model]
        choice=max(candidates,key=lambda r:r["score"]) if candidates else None
        family_choices[model]=choice
    for model,choice in family_choices.items():
        family_signals[model]=outer(choice) if choice is not None else []
    signals=family_signals[best["model"]] if best is not None else []
    return {"fold":fold.fold_id,"choice":best,"family_choices":family_choices,"trials":trials,
            "signals":signals,"family_signals":family_signals,"baseline":baseline}

def baseline_signals(symbol,paths,funding,folds):
    require({p.stem for p in paths}=={str(f.fold_id) for f in folds},"Missing frozen baseline folds "+symbol)
    result=[]
    for f in folds:
        p=next(p for p in paths if p.stem==str(f.fold_id))
        v=pl.read_parquet(p).sort("decision_time")
        require(v["symbol"].unique().to_list()==[symbol] and v["decision_time"].n_unique()==v.height and v["decision_time"].min()>=f.test_start and v["label_end"].max()<f.test_end,"Frozen prediction scope")
        require(((v["entry_time"]-v["decision_time"])==15*MIN).all() and ((v["label_end"]-v["entry_time"])==4*H).all(),"Frozen execution mismatch")
        p=v["prediction"].to_numpy()
        require(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),"Invalid frozen probabilities")
        v=v.with_columns(pl.Series("side",np.where(p>=.7,1,np.where(p<=.3,-1,0))),pl.Series("p_long",p),pl.Series("p_short",1-p)).filter(pl.col("side")!=0)
        if not v.height:continue
        e,z,ep,xp,side=[v[k].to_numpy() for k in ("entry_time","label_end","entry_price","exit_price","side")]
        net=[costs(e,z,ep,xp,side,funding,*c) for c in COSTS]
        opposite=[costs(e,z,ep,xp,-side,funding,*c) for c in COSTS]
        for i,t in enumerate(v.select("symbol","decision_time","entry_time","label_end","entry_price","exit_price","side","p_long","p_short").iter_rows(named=True)):
            t.update(fold=f.fold_id,tf="1h",horizon_hours=4,target="binary",model="logistic",C=1.,threshold=.7,exit_rule="fixed",
                     opposite_standard=float(opposite[0][i]) if np.isfinite(opposite[0][i]) else None,
                     opposite_stress=float(opposite[1][i]) if np.isfinite(opposite[1][i]) else None,standard=float(net[0][i]) if np.isfinite(net[0][i]) else None,
                     stress=float(net[1][i]) if np.isfinite(net[1][i]) else None)
            result.append(t)
    return result

def edge_stats(signals,which,first,last,bootstrap=False):
    """Fixed unit notional per outcome, NOT shared equity or portfolio return."""
    known=[t for t in signals if t[which] is not None]
    r=np.array([t[which] for t in known],dtype=float)
    row={"signals":len(signals),"scored_signals":len(known),
         "unknown_funding_signals":len(signals)-len(known),
         "coverage":len(known)/len(signals) if signals else None,
         "expectancy":None,"win_rate":None,"profit_factor":None,
         "sum_unit_notional_returns":None,
         "signals_per_30_days":len(signals)*30*DAY/(last-first)}
    if not len(r):return row
    require(np.isfinite(r).all(),"Invalid net returns")
    loss=-r[r<0].sum()
    row.update(expectancy=float(r.mean()),win_rate=float((r>0).mean()),
               profit_factor=float(r[r>0].sum()/loss) if loss>0 else None,
               sum_unit_notional_returns=float(r.sum()),
               no_losses=bool(loss==0))
    paired=[t for t in known if t.get("opposite_"+which) is not None]
    row["matched_direction_comparator"]={"pairs":len(paired),"chosen_minus_equal_random_direction_expectancy":None,"opposite_expectancy":None}
    if paired:
        delta=np.array([(t[which]-t["opposite_"+which])/2 for t in paired])
        row["matched_direction_comparator"].update(chosen_minus_equal_random_direction_expectancy=float(delta.mean()),opposite_expectancy=float(np.mean([t["opposite_"+which] for t in paired])))
    if bootstrap:
        # Calendar blocks preserve simultaneous cross-asset dependence.
        n=(last-first)//DAY
        days=np.array([(t["label_end"]-first)//DAY for t in known])
        require(np.all((days>=0)&(days<n)),"Outcome outside development calendar")
        daily=np.bincount(days,weights=r,minlength=n);count=np.bincount(days,minlength=n)
        row["bootstrap_nominal"]=[ci(daily,count,b) for b in (7,30)]
    return row

def evaluate_edge(signals,funds,first,last,fee,slip):
    which="standard" if (fee,slip)==COSTS[0] else "stress"
    # Reconcile vectorized cost arithmetic against the historical execution engine.
    for symbol,fund in funds.items():
        known=[t for t in signals if t["symbol"]==symbol and t[which] is not None]
        if not known:continue
        keys=("symbol","entry_time","label_end","entry_price","exit_price","side")
        df=pl.DataFrame([{k:t[k] for k in keys} for t in known])
        valid=fund[0].filter(pl.col("mark_price").is_finite().fill_null(False)&(pl.col("mark_price")>0).fill_null(False))
        scored=score_trades(df,fee,slip,valid,{symbol:[(A,B)]})
        require(scored.height==len(known) and np.allclose(scored["net_return"].to_numpy(),[t[which] for t in known],rtol=1e-10,atol=1e-12),"Edge cost reconciliation failed "+symbol)
    independent=nonoverlap(signals)
    def breakdown(rows,include_ci=False):
        return {"overall":edge_stats(rows,which,first,last,include_ci),
                "per_symbol":{symbol:edge_stats([t for t in rows if t["symbol"]==symbol],which,first,last) for symbol in funds},
                "per_side":{name:edge_stats([t for t in rows if t["side"]==side],which,first,last) for name,side in (("long",1),("short",-1))},
                "per_fold":{str(f.fold_id):edge_stats([t for t in rows if t["fold"]==f.fold_id],which,f.test_start,f.test_end) for f in make_folds(A,B,180,30)}}
    all_stats=breakdown(signals);nonoverlap_stats=breakdown(independent,True)
    # Equal-asset summary complements pooled results, which weight frequent signals more.
    for stats in (all_stats,nonoverlap_stats):
        vals=[v["expectancy"] for v in stats["per_symbol"].values() if v["expectancy"] is not None]
        stats["equal_symbol_mean_expectancy"]=float(np.mean(vals)) if vals else None
        stats["symbols_with_scored_signals"]=len(vals)
    unknown=[{k:t[k] for k in ("symbol","decision_time","entry_time","label_end","side","fold")} for t in signals if t[which] is None]
    return {"status":"incomplete_funding" if unknown else "pass","all_opportunities":all_stats,
            "per_symbol_nonoverlap":nonoverlap_stats,"unknown_funding_outcomes":unknown,
            "note":"All opportunities retained; overlapping returns are not independent; known-only results conditional if any funding is missing; no allocation or shared equity."}

def memory_preflight(stage,additional_mib=512):
    """Single-worker incremental budget after imports; estimate, not a measured peak."""
    reserve_mib=512
    vm=psutil.virtual_memory()
    available=vm.available
    required=(additional_mib+reserve_mib)*1024**2
    print(f"RAM {stage}: available {available//1024**2} MiB / estimated minimum {required//1024**2} MiB; process tree {tree_rss()//1024**2} MiB",flush=True)
    try:
        budget(1,per_worker_bytes=additional_mib*1024**2,reserve_bytes=reserve_mib*1024**2,threads_per_worker=1)
    except MemoryError as e:
        raise MemoryError(f"{stage}: {available//1024**2} MiB available; estimated minimum {required//1024**2} MiB. Close unused apps and resume the same command. Completed folds remain cached.") from e

def run(assets):
    import portalocker
    started=time.monotonic()
    root=ROOT/"research_v2";root.mkdir(parents=True,exist_ok=True)
    out={"protocol":SPEC,"assets_requested":assets,"scope":"Nested historical signal-edge search, not independent holdout. No shared-wallet allocation, capital limits or live orders. Five-asset default is a readiness pilot, not a survivorship-free universe.","assets":[],"edge_results":[]}
    report=ROOT/"reports"/"research_batch.zip";report.parent.mkdir(parents=True,exist_ok=True)
    signal_ledger={};fold_artifacts=[]
    with portalocker.Lock(str(root/"coordinator.lock"),timeout=0):
        try:
            memory_preflight("startup")
            out["memory_policy"]={"workers":1,"startup_and_dataset_additional_mib":512,"new_fold_additional_mib":256,"reserve_mib":512,"estimated_not_measured":True}
            gate=read_report("45_reproducibility.zip")
            require(gate["status"]=="pass","Frozen candidate reproducibility gate failed")
            folder,=list((ROOT/"experiments").glob("6823506d*"))
            old=read_report("25_multi_asset_transfer.zip");extended=read_report("39_extended_transfer.zip")
            oldrows={v["symbol"]:v for v in old["results"]+extended["results"] if v["status"]=="pass"}
            folds=make_folds(A,B,180,30)
            require(len(folds)==42,"Unexpected folds")
            first,last=folds[0].test_start,folds[-1].test_end
            out.update(first=first,last=last)
            code=digest([Path(n) for n in ["research_engine.py","data_engine.py","feature_engine.py","validation_protocol.py","execution_engine.py","experiment_store.py","baseline_engine.py","funding_repair_engine.py","robustness_engine.py"]])
            # Keep selection identity independent of final report presentation.
            model_code=hashlib.sha256((digest([Path(n) for n in ["data_engine.py","feature_engine.py","validation_protocol.py","execution_engine.py","funding_repair_engine.py","robustness_engine.py"]])+"".join(inspect.getsource(f) for f in (base_arrays,build,costs,predict,opportunities,nonoverlap,summarize_inner,inner_score,inner_outcomes,rule_signals,inner_masks,search_fold,load_funding))).encode()).hexdigest()
            builder_code=hashlib.sha256((digest([Path("data_engine.py"),Path("feature_engine.py")])+inspect.getsource(base_arrays)).encode()).hexdigest()
            search_spec=SPEC
            out.update(code_hash=code,model_code_hash=model_code)
            all_signals=[];baseline=[];funds={};families={m:[] for m in (*SPEC["models"],"rule")}
            for symbol in assets:
                row={"symbol":symbol,"folds":[]};out["assets"].append(row)
                try:
                    raw=monthly(ROOT/"futures"/symbol/"1m")
                    require(len(raw)==48,"Missing raw partitions")
                    sources=raw+monthly(ROOT/"funding"/symbol)+monthly(ROOT/"mark"/symbol)
                    if symbol=="BTCUSDT":
                        p=ROOT/"funding_repairs"/"BTCUSDT.json"
                        if p.exists():sources.append(p)
                        paths=sorted((folder/"predictions").glob("*.parquet"))
                    elif symbol in oldrows:paths=sorted((ROOT/"multi_asset"/oldrows[symbol]["experiment"]/"predictions").glob("*.parquet"))
                    else:paths=[]
                    if symbol=="BTCUSDT" or symbol in oldrows:require(paths,"Frozen baseline cache missing "+symbol)
                    fit_baseline=not paths
                    row["baseline_mode"]="new_fixed_fold_fit" if fit_baseline else "frozen_cache"
                    raw_hash=digest(raw)
                    funding=load_funding(symbol)
                    funding_hash=hashlib.sha256(funding[0].write_csv().encode()).hexdigest()
                    row.update(raw_hash=raw_hash,funding_ledger_hash=funding_hash,source_paths=[str(p) for p in sources],
                               baseline_prediction_hash=digest(paths) if paths else None)
                    key={"symbol":symbol,"protocol":search_spec,"raw":raw_hash,"funding":funding_hash,
                         "fit_baseline":fit_baseline,"code":model_code,"numpy":np.__version__,
                         "polars":pl.__version__,"sklearn":sklearn.__version__}
                    row["identity"]=identity(key)
                    row["unknown_funding_events"]=int(funding[3][-1])
                    row["unknown_funding_event_times"]=funding[1][np.diff(funding[3])>0].tolist()
                    cache_id=identity({"raw":raw_hash,"builder":builder_code,"polars":pl.__version__,"numpy":np.__version__,"TFS":TFS,"features":COLS,"horizons":HORIZONS})
                    completed=root/"folds"/row["identity"]
                    ds=None
                    if not all((completed/(str(f.fold_id)+".json")).exists() for f in folds):
                        gc.collect();memory_preflight(symbol+" dataset")
                        ds=build(symbol,raw,funding,root/"datasets"/cache_id)
                    signals=[];base=[];family_rows={m:[] for m in families}
                    for fold in folds:
                        fold_path=completed/(str(fold.fold_id)+".json")
                        if not fold_path.exists():
                            # Do not interrupt a fit; checkpoint boundary is safe for resume.
                            if psutil.virtual_memory().available<768*1024**2:
                                gc.collect();memory_preflight(symbol+" fold "+str(fold.fold_id),256)
                        result=checkpoint(root/"folds",key,fold.fold_id,lambda:search_fold(symbol,ds,fold,fit_baseline))
                        detail_name=f"folds/{symbol}/{fold.fold_id}.json"
                        fold_artifacts.append((fold_path,detail_name))
                        signals.extend(result["signals"]);base.extend(result["baseline"])
                        for m in families:family_rows[m].extend(result["family_signals"][m])
                        row["folds"].append({**{k:v for k,v in result.items() if k not in ("signals","family_signals","baseline","trials")},"trial_count":len(result["trials"]),"detail_file":detail_name,"outer_signals":len(result["signals"]),"family_outer_signals":{m:len(v) for m,v in result["family_signals"].items()},"new_baseline_signals":len(result["baseline"])})
                        del result
                        print(symbol,"fold",fold.fold_id,"done",flush=True)
                    if not fit_baseline:base=baseline_signals(symbol,paths,funding,folds)
                    all_signals.extend(signals);baseline.extend(base);funds[symbol]=funding
                    for m in families:families[m].extend(family_rows[m])
                    row.update(status="pass",outer_signals=len(signals),baseline_signals=len(base),rss_bytes=tree_rss())
                    del ds
                except MemoryError:
                    row.update(status="blocked_memory",error=traceback.format_exc());print(symbol,"paused: insufficient RAM",flush=True)
                    out["resume_required"]=True
                    break
                except Exception:
                    row.update(status="fail",error=traceback.format_exc());print(symbol,"blocked",flush=True)
                finally:
                    ds=None
                    gc.collect()
            require(funds,"No asset completed; inspect per-symbol errors and resume")
            out["assets_completed"]=list(funds)
            signal_ledger={"fixed_binary":baseline,"nested_joint_search":all_signals,**{"nested_"+m:v for m,v in families.items()}}
            for name,signals in signal_ledger.items():
                for fee,slip in COSTS:
                    row={"strategy":name,"fee_bps":fee,"slippage_bps":slip};out["edge_results"].append(row)
                    try:row.update(evaluate_edge(signals,funds,first,last,fee,slip))
                    except Exception:row.update(status="blocked",error=traceback.format_exc())
                    print(name,fee,slip,row["status"],flush=True)
            out["status"]="pass" if len(funds)==len(assets) and all(r["status"]=="pass" for r in out["assets"]+out["edge_results"]) else "partial"
        except Exception:out.update(status="fail",error=traceback.format_exc())
        out["elapsed_seconds"]=time.monotonic()-started
        tmp=report.with_suffix(".tmp")
        with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:
            # Stream large details from existing checkpoints; do not retain all trial dicts
            # in RAM or construct a single large JSON string for signal exports.
            for path,name in fold_artifacts:z.write(path,name)
            for name,obj in (("metrics.json",out),("signals.json",signal_ledger)):
                with z.open(name,"w") as binary:
                    with io.TextIOWrapper(binary,encoding="utf-8") as stream:
                        json.dump(obj,stream,allow_nan=False)
        tmp.replace(report);print(report,out["status"],flush=True)
    return 0 if out["status"]=="pass" else 1

# M2: frozen specification transfer. No parameter search, no capital constraints.
TRANSFER_SPEC={"tf":"1h","horizon_hours":4,"delay_minutes":15,"threshold":.70,
               "C":1.,"seed":42,"features":COLS,"train_days":180,"test_days":30,
               "min_train_coverage":.90,"costs_bps":COSTS,
               "universe":"all configured crypto symbols, regardless of prior performance"}

def transfer_dataset(symbol,raw,path):
    """One hourly feature build. Retain decisions with missing future prices."""
    if path.exists():return pl.read_parquet(path)
    bars=[];points=[]
    for p in raw:
        f=pl.read_parquet(p).filter(pl.col("open_time").is_between(A,B,closed="left")).sort("open_time")
        if not f.height:continue
        t=f["open_time"].to_numpy()
        require(np.all(t%MIN==0) and np.all(np.diff(t)>0),"Invalid raw timestamps "+str(p))
        v=f["open"].to_numpy()
        require(np.isfinite(v).all() and (v>0).all(),"Invalid raw opens "+str(p))
        bars.append(aggregate(f,"1h"))
        points.append(f.filter(pl.col("open_time")%H==15*MIN).select("open_time","open"))
    require(bars and points,"No usable development candles")
    h=technical(pl.concat(bars).sort("open_time"),H)
    p=pl.concat(points).sort("open_time")
    require(p["open_time"].n_unique()==p.height,"Duplicate execution timestamps")
    h=h.filter(pl.all_horizontal([pl.col(c).is_finite().fill_null(False) for c in BASE_FEATURES]))
    h=h.with_columns(pl.lit(symbol).alias("symbol"),(pl.col("decision_time")+15*MIN).alias("entry_time"),
                     (pl.col("decision_time")+15*MIN+4*H).alias("label_end"))
    for k,n in (("entry_time","entry_price"),("label_end","exit_price")):
        h=h.join(p.rename({"open_time":k,"open":n}),on=k,how="left",validate="m:1")
    h=h.select("symbol","decision_time","entry_time","label_end","entry_price","exit_price",*COLS).sort("decision_time")
    from data_engine import atomic_write
    atomic_write(h,path)
    return h

def transfer_fit_fold(frame,fold):
    # Train eligibility uses only completed historical outcomes. No minimum future test count.
    train=frame.filter((pl.col("decision_time")>=fold.train_start)&(pl.col("label_end")<fold.test_start)&
                       pl.col("entry_price").is_not_null()&pl.col("exit_price").is_not_null())
    test=frame.filter((pl.col("decision_time")>=fold.test_start)&(pl.col("label_end")<fold.test_end))
    expected=(fold.test_start-fold.train_start)//H-5
    row={"fold":fold.fold_id,"train_rows":train.height,"expected_train_rows":expected,
         "test_rows":test.height,"signals":[]}
    if train.height<TRANSFER_SPEC["min_train_coverage"]*expected:
        row["status"]="insufficient_historical_coverage";return row
    if not test.height:
        row["status"]="no_observed_decisions";return row
    y=(train["exit_price"].to_numpy()>train["entry_price"].to_numpy()).astype(int)
    if len(np.unique(y))!=2:
        row["status"]="single_training_class";return row
    with threadpool_limits(limits=1):
        scale=StandardScaler()
        x=scale.fit_transform(train.select(COLS).to_numpy())
        p=LogisticRegression(C=1.,max_iter=500,random_state=42).fit(x,y).predict_proba(scale.transform(test.select(COLS).to_numpy()))[:,1]
    row["status"]="pass"
    row["signals"]=transfer_extract(test,p,fold.fold_id)
    return row

def transfer_extract(frame,p,fold_id):
    require(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),"Invalid transfer probabilities")
    v=frame.select("symbol","decision_time","entry_time","label_end","entry_price","exit_price").with_columns(
        pl.Series("p_long",p),pl.Series("side",np.where(p>=.7,1,np.where(p<=.3,-1,0)))).filter(pl.col("side")!=0)
    return [{**t,"fold":fold_id} for t in v.iter_rows(named=True)]

def transfer_cached(symbol,paths,folds):
    require({p.stem for p in paths}=={str(f.fold_id) for f in folds},"Incomplete frozen predictions")
    by_id={p.stem:p for p in paths};rows=[];signals=[]
    for f in folds:
        v=pl.read_parquet(by_id[str(f.fold_id)]).sort("decision_time")
        require(v.height and v["symbol"].unique().to_list()==[symbol] and v["decision_time"].n_unique()==v.height,"Cached symbol/keys mismatch")
        require(v["decision_time"].min()>=f.test_start and v["label_end"].max()<f.test_end and
                ((v["entry_time"]-v["decision_time"])==15*MIN).all() and
                ((v["label_end"]-v["entry_time"])==4*H).all(),"Cached chronology mismatch")
        t=transfer_extract(v,v["prediction"].to_numpy(),f.fold_id)
        signals.extend(t);rows.append({"fold":f.fold_id,"status":"cached","test_rows":v.height,"signals":len(t)})
    return rows,signals

def transfer_funding(symbol):
    """Verify local 8h schedule spans, including shorter listed histories.
    Never extrapolate funding beyond verified spans, or replace missing rates/marks with zero.
    """
    if symbol=="BTCUSDT":
        return load_funding(symbol),[(A,B)]
    parts=[]
    for p in monthly(ROOT/"funding"/symbol):
        x=pl.read_parquet(p).select("event_time","funding_rate","funding_interval_hours")
        mp=ROOT/"mark"/symbol/p.name
        if mp.exists():
            m=pl.read_parquet(mp,columns=["event_time","open"]).rename({"event_time":"minute","open":"mark_price"})
            x=x.with_columns((pl.col("event_time")//MIN*MIN).alias("minute")).join(m,on="minute",how="left",validate="m:1")
        else:x=x.with_columns(pl.lit(None,dtype=pl.Float64).alias("mark_price"))
        parts.append(x.select("event_time","funding_rate","funding_interval_hours","mark_price"))
    if not parts:return None,[]
    f=pl.concat(parts).filter(pl.col("event_time").is_between(A,B,closed="left")).sort("event_time")
    if not f.height:return None,[]
    ts=f["event_time"].to_numpy()
    require(len(np.unique(ts))==len(ts),"Duplicate funding events")
    grid=((ts+MIN//2)//(8*H))*(8*H)
    good_grid=(np.abs(ts-grid)<=MIN)&(f["funding_interval_hours"].fill_null(-1).to_numpy()==8)
    spans=[];start=None
    for i in range(len(ts)-1):
        if good_grid[i] and good_grid[i+1] and grid[i+1]-grid[i]==8*H:
            if start is None:start=int(ts[i])
        elif start is not None:
            spans.append((start,int(ts[i])));start=None
    if start is not None:spans.append((start,int(ts[-1])))
    # Same complete-calendar convention as the original audited 8h engine.
    if good_grid.all() and np.array_equal(grid,np.arange(A,B,8*H)):spans=[(A,B)]
    rate=f["funding_rate"].cast(pl.Float64).to_numpy();price=f["mark_price"].cast(pl.Float64).to_numpy()
    good=np.isfinite(rate)&np.isfinite(price)&(price>0)
    f=f.with_columns(pl.lit(symbol).alias("symbol")).select("symbol","event_time","funding_rate","mark_price")
    return (f,ts,np.r_[0.,np.cumsum(np.where(good,rate*price,0.))],np.r_[0,np.cumsum(~good)]),spans

def transfer_score(signals,fund,spans):
    # Preserve every signal; distinguish unknown price, schedule, and settlement mark.
    for t in signals:
        t.update(standard=None,stress=None,fee_slip_standard=None,fee_slip_stress=None,
                 outcome_status="unknown_price")
    known=[t for t in signals if all(t[k] is not None and np.isfinite(t[k]) and t[k]>0 for k in ("entry_price","exit_price"))]
    if not known:return
    e,z,ep,xp,side=[np.array([t[k] for t in known]) for k in ("entry_time","label_end","entry_price","exit_price","side")]
    covered=np.array([any(a<=x and y<=b for a,b in spans) for x,y in zip(e,z)],dtype=bool)
    for (fee,slip),which in zip(COSTS,("standard","stress")):
        u=ep*(1+side*slip/10000);v=xp*(1-side*slip/10000)
        pre=side*(v-u)/u-fee/10000*(u+v)/u
        net=costs(e,z,ep,xp,side,fund,fee,slip) if fund is not None else np.full(len(e),np.nan)
        net=np.where(covered,net,np.nan)
        for i,t in enumerate(known):
            t["fee_slip_"+which]=float(pre[i])
            t[which]=float(net[i]) if np.isfinite(net[i]) else None
            t["outcome_status"]="scored" if np.isfinite(net[i]) else ("unknown_funding_event" if covered[i] else "unverified_funding_schedule")
        valid=[t for t in known if t[which] is not None]
        if valid:
            keys=("symbol","entry_time","label_end","entry_price","exit_price","side")
            frame=pl.DataFrame([{k:t[k] for k in keys} for t in valid])
            f=fund[0].filter(pl.col("funding_rate").is_finite().fill_null(False)&pl.col("mark_price").is_finite().fill_null(False)&(pl.col("mark_price")>0).fill_null(False))
            scored=score_trades(frame,fee,slip,f,{valid[0]["symbol"]:spans})
            require(np.allclose(scored["net_return"].to_numpy(),[t[which] for t in valid],atol=1e-12,rtol=1e-10),"Transfer cost reconciliation")

def transfer_summary(signals,first,last,bootstrap=False):
    def stats(rows,which,ci_on):
        v=edge_stats(rows,which,first,last,ci_on)
        v["unknown_outcomes"]=v.pop("unknown_funding_signals")
        v.pop("matched_direction_comparator",None)
        return v
    result={}
    for name,rows in (("all_opportunities",signals),("per_symbol_nonoverlap",nonoverlap(signals))):
        counts={}
        for t in rows:counts[t["outcome_status"]]=counts.get(t["outcome_status"],0)+1
        result[name]={"outcome_counts":counts,
                      "standard":stats(rows,"standard",bootstrap and name=="per_symbol_nonoverlap"),
                      "stress":stats(rows,"stress",bootstrap and name=="per_symbol_nonoverlap"),
                      "fee_slip_only_standard":stats(rows,"fee_slip_standard",False),
                      "fee_slip_only_stress":stats(rows,"fee_slip_stress",False)}
    return result

def run_transfer(assets):
    import portalocker
    started=time.monotonic()
    root=ROOT/"research_v2"/"transfer";root.mkdir(parents=True,exist_ok=True)
    report=ROOT/"reports"/"research_transfer.zip";report.parent.mkdir(parents=True,exist_ok=True)
    out={"protocol":TRANSFER_SPEC,"assets_requested":assets,"assets":[],
         "scope":"Fixed specification development transfer; historical universe purposive, not survivorship-free. Previously screened negative assets retained. New fits use observed historical coverage, not verified listing dates. No capital constraints, tuning, holdout or downloads. Fee/slippage-only outcomes are NOT funding-adjusted.",
         "cohorts":{}}
    all_signals=[]
    with portalocker.Lock(str(root/"coordinator.lock"),timeout=0):
        try:
            memory_preflight("transfer startup")
            folder,=list((ROOT/"experiments").glob("6823506d*"))
            require(json.loads((folder/"result.json").read_text())["params"]["C"]==1.,"Frozen C differs from transfer specification")
            prior=read_report("research_batch.zip")
            require(set(prior["assets_completed"])==set(ASSETS),"Five-asset batch incomplete")
            prior_rows={r["symbol"]:r for r in prior["assets"]}
            oldrows={}
            attempted=set()
            for name in ("25_multi_asset_transfer.zip","39_extended_transfer.zip"):
                for r in read_report(name)["results"]:
                    attempted.add(r["symbol"])
                    if r["status"]=="pass":oldrows[r["symbol"]]=r
            folds=make_folds(A,B,180,30);first,last=folds[0].test_start,folds[-1].test_end
            out.update(first=first,last=last)
            source_code=hashlib.sha256((digest([Path("data_engine.py"),Path("feature_engine.py"),Path("validation_protocol.py")])+
                                      "".join(inspect.getsource(f) for f in (transfer_dataset,transfer_fit_fold,transfer_extract))).encode()).hexdigest()
            for symbol in assets:
                row={"symbol":symbol,"cohort":"pilot" if symbol in ASSETS else ("prior_screened" if symbol in attempted else "additional_universe")}
                out["assets"].append(row)
                try:
                    raw=monthly(ROOT/"futures"/symbol/"1m")
                    if not raw:
                        row["status"]="no_development_raw";continue
                    raw_hash=digest(raw);row["raw_hash"]=raw_hash
                    paths=[]
                    if symbol=="BTCUSDT" and prior_rows[symbol]["raw_hash"]==raw_hash:
                        paths=sorted((folder/"predictions").glob("*.parquet"))
                    elif symbol in oldrows and oldrows[symbol].get("source")==raw_hash:
                        paths=sorted((ROOT/"multi_asset"/oldrows[symbol]["experiment"]/"predictions").glob("*.parquet"))
                    if paths and {p.stem for p in paths}=={str(f.fold_id) for f in folds}:
                        ph=digest(paths)
                        if symbol in prior_rows:
                            require(ph==prior_rows[symbol]["baseline_prediction_hash"],"Pilot prediction hash changed")
                        row.update(mode="frozen_cache",prediction_hash=ph)
                        row["folds"],signals=transfer_cached(symbol,paths,folds)
                    else:
                        row["mode"]="fixed_refit"
                        key={"symbol":symbol,"raw":raw_hash,"spec":TRANSFER_SPEC,"code":source_code,
                             "numpy":np.__version__,"polars":pl.__version__,"sklearn":sklearn.__version__}
                        cache_id=identity(key);row["identity"]=cache_id
                        cache=root/"datasets"/cache_id/"dataset.parquet"
                        completed=root/"folds"/cache_id
                        frame=None;signals=[];row["folds"]=[]
                        for fold in folds:
                            if not (completed/(str(fold.fold_id)+".json")).exists() and frame is None:
                                memory_preflight(symbol+" transfer dataset")
                                frame=transfer_dataset(symbol,raw,cache)
                            v=checkpoint(root/"folds",key,fold.fold_id,lambda:transfer_fit_fold(frame,fold))
                            signals.extend(v["signals"])
                            row["folds"].append({**{k:x for k,x in v.items() if k!="signals"},"signals":len(v["signals"])})
                        frame=None
                    row["signals"]=len(signals)
                    row["eligible_folds"]=sum(f["status"] in ("pass","cached") for f in row["folds"])
                    fund,spans=transfer_funding(symbol)
                    row["verified_funding_spans"]=spans
                    transfer_score(signals,fund,spans)
                    row["results"]=transfer_summary(signals,first,last)
                    row["status"]="pass" if all(t["outcome_status"]=="scored" for t in signals) else "partial_outcomes"
                    if not row["eligible_folds"]:row["status"]="no_eligible_folds"
                    all_signals.extend(signals)
                    print(symbol,row["status"],len(signals),"signals",flush=True)
                except MemoryError:
                    row.update(status="blocked_memory",error=traceback.format_exc());break
                except Exception:
                    row.update(status="fail",error=traceback.format_exc());print(symbol,"blocked",flush=True)
                finally:
                    frame=None;gc.collect()
            for name in ("pilot","prior_screened","additional_universe","outside_pilot","all"):
                symbols={r["symbol"] for r in out["assets"] if name=="all" or r["cohort"]==name or (name=="outside_pilot" and r["cohort"]!="pilot")}
                v=[t for t in all_signals if t["symbol"] in symbols]
                out["cohorts"][name]={"symbols":sorted(symbols),"results":transfer_summary(v,first,last,True)}
            out["status"]="pass" if len(out["assets"])==len(assets) and all(r["status"]=="pass" for r in out["assets"]) else "partial"
        except Exception:out.update(status="fail",error=traceback.format_exc())
        out["elapsed_seconds"]=time.monotonic()-started
        tmp=report.with_suffix(".tmp")
        with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:
            for name,obj in (("metrics.json",out),("signals.json",all_signals)):
                with z.open(name,"w") as binary:
                    with io.TextIOWrapper(binary,encoding="utf-8") as stream:
                        json.dump(obj,stream,allow_nan=False)
        tmp.replace(report);print(report,out["status"],flush=True)
    return 0 if out["status"]=="pass" else 1

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--symbols",nargs="+",choices=CRYPTO_SYMBOLS,default=ASSETS)
    a=p.parse_args()
    raise SystemExit(run(list(dict.fromkeys(a.symbols))))
