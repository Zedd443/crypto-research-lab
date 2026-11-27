"""Integrated, resumable multi-asset research batch; no live orders or holdout access.
Historical experiment scripts remain intact. One batch report per protocol.
"""
import os
os.environ.setdefault("POLARS_MAX_THREADS","1")
import json, zipfile, traceback, hashlib, time, inspect
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
      "holdout_excluded":True}

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

def inner_score(signals,first,last):
    # Avoid counting repeated overlapping outcomes as independent evidence.
    # No quota on frequency and no cross-symbol constraint.
    selected=nonoverlap(signals)
    n=len(selected);unknown=sum(t["standard"] is None or t["stress"] is None for t in selected)
    row={"signals":len(signals),"nonoverlap_signals":n,"unknown_funding_trades":unknown,"score":None}
    if not n or unknown:
        row["reason"]="no_signals" if not n else "unknown_funding"
        return row
    primary=np.array([t["standard"] for t in selected])
    stress=np.array([t["stress"] for t in selected])
    require(all(first<=t["decision_time"]<t["label_end"]<last for t in selected),"Inner signal outside validation")
    row.update(primary_expectancy=float(primary.mean()),stress_expectancy=float(stress.mean()),
               primary_pnl=float(primary.sum()),stress_pnl=float(stress.sum()))
    if primary.mean()>0:
        row["score"]=float(primary.sum()/(n+SPEC["edge_shrinkage_zero_prior_trades"]))
    else:row["reason"]="nonpositive_primary_expectancy"
    return row

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
        candidate=(ds["d"]>=lo)&(ds["z"]<hi)
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
                signals=[];diagnostics=[]
                for val,p,side,lo,hi,total in windows:
                    v=opportunities(ds,val,p,th,rule,side)
                    signals.extend(v)
                    d=inner_score(v,lo,hi)
                    diagnostics.append({"start":lo,"end":hi,"eligible_rows":int(val.sum()),
                                        "candidate_rows":total,**d})
                row=inner_score(signals,first,fold.test_start)
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
    if not eligible:return {"fold":fold.fold_id,"choice":None,"trials":trials,"signals":[],"baseline":baseline}
    best=max(eligible,key=lambda r:r["score"])
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
    return {"fold":fold.fold_id,"choice":best,"trials":trials,"signals":signals,"baseline":baseline}

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
        df=pl.DataFrame(known).select("symbol","entry_time","label_end","entry_price","exit_price","side")
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

def run(assets):
    import portalocker
    started=time.monotonic()
    budget(1,per_worker_bytes=1024**3,threads_per_worker=1)
    root=ROOT/"research_v2";root.mkdir(parents=True,exist_ok=True)
    out={"protocol":SPEC,"assets_requested":assets,"scope":"Nested historical signal-edge search, not independent holdout. No shared-wallet allocation, capital limits or live orders. Five-asset default is a readiness pilot, not a survivorship-free universe.","assets":[],"edge_results":[]}
    report=ROOT/"reports"/"research_batch.zip";report.parent.mkdir(parents=True,exist_ok=True)
    signal_ledger={}
    with portalocker.Lock(str(root/"coordinator.lock"),timeout=0):
        try:
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
            model_code=hashlib.sha256((digest([Path(n) for n in ["data_engine.py","feature_engine.py","validation_protocol.py","execution_engine.py","funding_repair_engine.py","robustness_engine.py"]])+"".join(inspect.getsource(f) for f in (base_arrays,build,costs,predict,opportunities,nonoverlap,inner_score,rule_signals,inner_masks,search_fold,load_funding))).encode()).hexdigest()
            builder_code=hashlib.sha256((digest([Path("data_engine.py"),Path("feature_engine.py")])+inspect.getsource(base_arrays)).encode()).hexdigest()
            search_spec=SPEC
            out.update(code_hash=code,model_code_hash=model_code)
            all_signals=[];baseline=[];funds={}
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
                    ds=None if all((completed/(str(f.fold_id)+".json")).exists() for f in folds) else build(symbol,raw,funding,root/"datasets"/cache_id)
                    signals=[];base=[]
                    for fold in folds:
                        result=checkpoint(root/"folds",key,fold.fold_id,lambda:search_fold(symbol,ds,fold,fit_baseline))
                        signals.extend(result["signals"]);base.extend(result["baseline"])
                        row["folds"].append({**{k:v for k,v in result.items() if k not in ("signals","baseline")},"outer_signals":len(result["signals"]),"new_baseline_signals":len(result["baseline"])})
                        print(symbol,"fold",fold.fold_id,"done",flush=True)
                    if not fit_baseline:base=baseline_signals(symbol,paths,funding,folds)
                    all_signals.extend(signals);baseline.extend(base);funds[symbol]=funding
                    row.update(status="pass",outer_signals=len(signals),baseline_signals=len(base),rss_bytes=tree_rss())
                    del ds
                except Exception:
                    row.update(status="fail",error=traceback.format_exc());print(symbol,"blocked",flush=True)
                finally:
                    ds=None
            require(funds,"No asset completed")
            out["assets_completed"]=list(funds)
            signal_ledger={"fixed_binary":baseline,"nested_joint_search":all_signals}
            for name,signals in signal_ledger.items():
                for fee,slip in COSTS:
                    row={"strategy":name,"fee_bps":fee,"slippage_bps":slip};out["edge_results"].append(row)
                    try:row.update(evaluate_edge(signals,funds,first,last,fee,slip))
                    except Exception:row.update(status="blocked",error=traceback.format_exc())
                    print(name,fee,slip,row["status"],flush=True)
            out["status"]="pass" if all(r["status"]=="pass" for r in out["assets"]+out["edge_results"]) else "partial"
        except Exception:out.update(status="fail",error=traceback.format_exc())
        out["elapsed_seconds"]=time.monotonic()-started
        tmp=report.with_suffix(".tmp")
        with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:
            z.writestr("metrics.json",json.dumps(out,allow_nan=False))
            z.writestr("signals.json",json.dumps(signal_ledger,allow_nan=False))
        tmp.replace(report);print(report,out["status"],flush=True)
    return 0 if out["status"]=="pass" else 1

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--symbols",nargs="+",choices=CRYPTO_SYMBOLS,default=ASSETS)
    a=p.parse_args()
    raise SystemExit(run(list(dict.fromkeys(a.symbols))))
