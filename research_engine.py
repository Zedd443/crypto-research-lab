"""Integrated, resumable multi-asset research batch; no live orders or holdout access.
Historical experiment scripts remain intact. One batch report per protocol.
"""
import os
os.environ.setdefault("POLARS_MAX_THREADS","1")
import json, zipfile, traceback, hashlib, time
from pathlib import Path
from collections import Counter
import numpy as np
import polars as pl
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from baseline_engine import digest
from feature_engine import technical, BASE_FEATURES
from validation_protocol import Fold, make_folds
from experiment_store import identity, checkpoint
from robustness_engine import A,B,DAY,ledger
from funding_repair_engine import apply
from data_engine import aggregate
from execution_engine import score_trades
from portfolio_engine import mark_window
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
      "targets":["binary","cost_aware"],"coarse_C":1.,"refine_C":[.1,10.],
      "refine_families":1,"exit_rules":RULES,"features":COLS,
      "delay_minutes":15,"train_days":180,"inner_fit_days":120,"inner_validation_days":60,
      "outer_test_days":30,"seed":42,"min_inner_trades":5,
      "objective":"positive primary and stress P&L; maximize stress daily fixed-notional Sharpe; cash if none",
      "costs_bps":COSTS,"wallet":{"max_positions":5,"position_fraction":.2,
      "entry_gross_cap":.95,"min_position_fraction":.05},
      "execution":"decision then 15m delay; allocate at entry; exits before entries; one position per symbol",
      "funding":"strict 8h schedule; no unknown funding imputation; block evaluation if selected trade affected",
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
            datasets[(tf,horizon)]={"d":d,"e":e,"z":z,"ep":ep,"xp":xp,"x":x,
                "binary":np.where(xp>ep,1,-1),"cost_aware":y,"known":np.isfinite(l)&np.isfinite(s),
                "variants":variants}
    return datasets

def predict(ds,target,C,train,test):
    # Only historical training rows with known realized funding enter either target.
    train=train&ds["known"]
    y=ds[target][train]
    expected=np.array([-1,1] if target=="binary" else [-1,0,1])
    if train.sum()<100 or test.sum()<30 or not np.array_equal(np.unique(y),expected):
        return None
    with threadpool_limits(limits=1):
        scale=StandardScaler()
        x=scale.fit_transform(ds["x"][train]);v=scale.transform(ds["x"][test])
        model=LogisticRegression(C=C,max_iter=500,random_state=42).fit(x,y)
        p=model.predict_proba(v)
    require(np.array_equal(model.classes_,expected) and np.isfinite(p).all(),"Invalid predictions")
    return p[:,0],p[:,-1]

def opportunities(ds,test,p,threshold,rule):
    ix=np.flatnonzero(test)
    side=np.where(p[1]>=threshold,1,np.where(p[0]>=threshold,-1,0))
    hit=np.flatnonzero(side)
    result=[]
    for j in hit:
        i=ix[j];v=int(side[j]);end,xp,standard,stress=ds["variants"][(v,rule)]
        result.append({"decision_time":int(ds["d"][i]),"entry_time":int(ds["e"][i]),
            "label_end":int(end[i]),"entry_price":float(ds["ep"][i]),"exit_price":float(xp[i]),"side":v,
            "standard":float(standard[i]) if np.isfinite(standard[i]) else None,
            "stress":float(stress[i]) if np.isfinite(stress[i]) else None})
    return result

def inner_score(signals,first,last):
    free=-1;selected=[]
    for t in signals:
        if t["entry_time"]>=free:selected.append(t);free=t["label_end"]
    n=len(selected);unknown=sum(t["standard"] is None or t["stress"] is None for t in selected)
    row={"signals":len(signals),"trades":n,"unknown_funding_trades":unknown,"score":None}
    if not n or unknown:return row
    primary=np.array([t["standard"] for t in selected])
    stress=np.array([t["stress"] for t in selected])
    days=np.array([(t["label_end"]-first)//DAY for t in selected])
    require(np.all((days>=0)&(days<(last-first)//DAY)),"Inner exit outside validation")
    daily=np.bincount(days,weights=stress,minlength=(last-first)//DAY)
    row.update(primary_pnl=float(primary.sum()),stress_pnl=float(stress.sum()))
    sd=daily.std(ddof=1)
    if n>=SPEC["min_inner_trades"] and primary.sum()>0 and stress.sum()>0 and sd>0:
        row["score"]=float(daily.mean()/sd*np.sqrt(365))
    return row

def search_fold(symbol,ds_all,fold,fit_baseline=False):
    cut=fold.train_start+120*DAY
    trials=[];baseline=[]
    if fit_baseline:
        ds=ds_all[("1h",4)]
        train=(ds["d"]>=fold.train_start)&(ds["z"]<fold.test_start)
        test=(ds["d"]>=fold.test_start)&(ds["z"]<fold.test_end)
        p=predict(ds,"binary",1.,train,test)
        require(p is not None,"Cannot fit fixed baseline")
        baseline=opportunities(ds,test,p,.7,"fixed")
        for t in baseline:t.update(symbol=symbol,priority=0.,fold=fold.fold_id)
    def evaluate(tf,h,target,C):
        ds=ds_all[(tf,h)]
        train=(ds["d"]>=fold.train_start)&(ds["z"]<cut)
        val=(ds["d"]>=cut)&(ds["z"]<fold.test_start)
        p=predict(ds,target,C,train,val)
        if p is None:
            trials.append({"tf":tf,"horizon":h,"target":target,"C":C,"score":None,"reason":"insufficient rows/classes"})
            return
        for rule in RULES:
            for th in THRESHOLDS:
                row=inner_score(opportunities(ds,val,p,th,rule),cut,fold.test_start)
                trials.append({"tf":tf,"horizon":h,"target":target,"C":C,"exit":rule,"threshold":th,**row})
    for tf,h in ds_all:
        for target in SPEC["targets"]:evaluate(tf,h,target,1.)
    eligible=[r for r in trials if r["score"] is not None]
    if eligible:
        best=max(eligible,key=lambda r:r["score"])
        for C in SPEC["refine_C"]:evaluate(best["tf"],best["horizon"],best["target"],C)
    eligible=[r for r in trials if r["score"] is not None]
    if not eligible:return {"fold":fold.fold_id,"choice":None,"trials":trials,"signals":[],"baseline":baseline}
    best=max(eligible,key=lambda r:r["score"])
    ds=ds_all[(best["tf"],best["horizon"])]
    train=(ds["d"]>=fold.train_start)&(ds["z"]<fold.test_start)
    test=(ds["d"]>=fold.test_start)&(ds["z"]<fold.test_end)
    p=predict(ds,best["target"],best["C"],train,test)
    require(p is not None,"Selected configuration cannot refit")
    signals=opportunities(ds,test,p,best["threshold"],best["exit"])
    for t in signals:t.update(symbol=symbol,priority=best["score"],fold=fold.fold_id)
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
        v=v.with_columns(pl.Series("side",np.where(p>=.7,1,np.where(p<=.3,-1,0)))).filter(pl.col("side")!=0)
        if not v.height:continue
        e,z,ep,xp,side=[v[k].to_numpy() for k in ("entry_time","label_end","entry_price","exit_price","side")]
        net=[costs(e,z,ep,xp,side,funding,*c) for c in COSTS]
        for i,t in enumerate(v.select("symbol","decision_time","entry_time","label_end","entry_price","exit_price","side").iter_rows(named=True)):
            t.update(priority=0.,fold=f.fold_id,standard=float(net[0][i]) if np.isfinite(net[0][i]) else None,
                     stress=float(net[1][i]) if np.isfinite(net[1][i]) else None)
            result.append(t)
    return result

def wallet(signals,funds,first,last,fee,slip,out):
    """One unlevered shared wallet; entry-time reservation, minute-open MTM."""
    policy=SPEC["wallet"];which="standard" if (fee,slip)==COSTS[0] else "stress"
    ordered=sorted(signals,key=lambda t:(t["entry_time"],-t["priority"],t["symbol"]))
    active={};balance=1.;peak=1.;dd=0.;daily={};records=[];reject=Counter()
    max_open=0;max_reserve=0.;max_mark_gross=0.;active_minutes=0;max_opportunities=0
    i=0;t=first
    def equity(t):
        return balance+sum(pos["floating"][(t-pos["entry_time"])//MIN] for pos in active.values())
    def snapshot(t):
        nonlocal peak,dd,max_reserve,max_mark_gross
        eq=equity(t)
        require(np.isfinite(eq) and eq>0,"Nonpositive or invalid equity")
        peak=max(peak,eq);dd=max(dd,1-eq/peak)
        max_reserve=max(max_reserve,sum(p["notional"] for p in active.values())/eq)
        max_mark_gross=max(max_mark_gross,sum(p["quantity"]*p["marks"][(t-p["entry_time"])//MIN] for p in active.values())/eq)
        daily[t//DAY]=eq
        return eq
    out["opportunities"]=len(ordered)
    while i<len(ordered) or active:
        if not active and i<len(ordered):t=ordered[i]["entry_time"]
        for symbol in list(active):
            p=active[symbol]
            if p["label_end"]==t:
                balance+=p["notional"]*p["net"];del active[symbol]
        eq=snapshot(t)
        j=i
        while i<len(ordered) and ordered[i]["entry_time"]==t:
            tr=ordered[i];i+=1;s=tr["symbol"]
            if s in active:reject["symbol_busy"]+=1;continue
            if len(active)>=policy["max_positions"]:reject["slots"]+=1;continue
            used=sum(p["notional"] for p in active.values())
            notional=min(eq*policy["position_fraction"],eq*policy["entry_gross_cap"]-used)
            if notional<eq*policy["min_position_fraction"]:reject["capital"]+=1;continue
            out["blocked_trade"]=tr
            require(tr[which] is not None,"Accepted trade crosses unknown funding mark")
            e,z=tr["entry_time"],tr["label_end"]
            m,absent=mark_window(e,z,s)
            require(m is not None and not absent,"Missing mark partition "+str(absent))
            ts=m["event_time"].to_numpy();px=m["open"].cast(pl.Float64).to_numpy()
            want=np.arange(e,z,MIN)
            if not np.array_equal(ts,want):
                out["missing_mark_minutes"]=np.setdiff1d(want,ts).tolist()
                raise ValueError("Incomplete intratrade mark grid")
            require(np.isfinite(px).all() and (px>0).all(),"Invalid intratrade marks")
            side=tr["side"];entry_fill=tr["entry_price"]*(1+side*slip/10000)
            q=notional/entry_fill
            f=funds[s][0].filter((pl.col("event_time")>e)&(pl.col("event_time")<=z))
            ft=f["event_time"].to_numpy()
            cashflow=f["funding_rate"].to_numpy()*f["mark_price"].to_numpy()*q*side
            require(np.isfinite(cashflow).all(),"Missing settlement price")
            cum=np.r_[0.,np.cumsum(cashflow)]
            floating=q*side*(px-entry_fill)-q*entry_fill*fee/10000-cum[np.searchsorted(ft,ts,side="right")]
            active[s]={**tr,"notional":notional,"quantity":q,"net":tr[which],"floating":floating,"marks":px}
            records.append({**tr,"notional":notional,"quantity":q,"net_return":tr[which]})
            out.pop("blocked_trade",None)
            max_open=max(max_open,len(active));eq=snapshot(t)
        max_opportunities=max(max_opportunities,i-j)
        if active:active_minutes+=1
        t+=MIN
    require(len(records)+sum(reject.values())==len(ordered),"Opportunity reconciliation")
    if records:
        df=pl.DataFrame(records).select("symbol","entry_time","label_end","entry_price","exit_price","side")
        fund=pl.concat([f[0].filter(pl.col("mark_price").is_finite().fill_null(False)&(pl.col("mark_price")>0).fill_null(False)) for f in funds.values()])
        scored=score_trades(df,fee,slip,fund,{s:[(A,B)] for s in funds})
        require(np.allclose(scored["net_return"].to_numpy(),[r["net_return"] for r in records],rtol=1e-10,atol=1e-12),"Shared wallet cost reconciliation")
        require(np.isclose(balance,1+sum(r["notional"]*r["net_return"] for r in records)),"Shared wallet cash reconciliation")
    curve=[];prev=1.
    for day in range(first//DAY,last//DAY):
        prev=daily.get(day,prev);curve.append(prev)
    r=np.diff(np.r_[1.,curve])/np.r_[1.,curve][:-1]
    sd=r.std(ddof=1)
    out.update(status="pass",trades=len(records),rejected=dict(reject),return_fraction=balance-1,
        max_mtm_drawdown=float(dd),daily_sharpe=float(r.mean()/sd*np.sqrt(365)) if sd>0 else None,
        max_open=max_open,peak_reserved_to_equity=float(max_reserve),peak_mark_gross_to_equity=float(max_mark_gross),
        max_simultaneous_opportunities=max_opportunities,exposure_any_position=active_minutes*MIN/(last-first),
        trades_per_symbol=dict(Counter(x["symbol"] for x in records)),daily_equity=curve,ledger=records)

def run(assets):
    import portalocker
    started=time.monotonic()
    budget(1,per_worker_bytes=1024**3,threads_per_worker=1)
    root=ROOT/"research_v2";root.mkdir(parents=True,exist_ok=True)
    out={"protocol":SPEC,"assets_requested":assets,"scope":"Nested historical development search, not independent holdout. No live orders. Five-asset default is a readiness pilot, not a survivorship-free universe. Entry notional cap is not Binance margin/liquidation modeling.","assets":[],"portfolios":[]}
    report=ROOT/"reports"/"research_batch.zip";report.parent.mkdir(parents=True,exist_ok=True)
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
            code=digest([Path(n) for n in ["research_engine.py","data_engine.py","feature_engine.py","validation_protocol.py","execution_engine.py","portfolio_engine.py","experiment_store.py","baseline_engine.py","funding_repair_engine.py","robustness_engine.py"]])
            all_signals=[];baseline=[];funds={}
            for symbol in assets:
                row={"symbol":symbol,"folds":[]};out["assets"].append(row)
                try:
                    raw=monthly(ROOT/"futures"/symbol/"1m")
                    require(len(raw)==48,"Missing raw partitions")
                    sources=raw+monthly(ROOT/"funding"/symbol)+monthly(ROOT/"mark"/symbol)
                    sources+=sorted(p for p in (ROOT/"mark_repairs"/symbol).glob("*") if p.is_file() and "2021-01"<=p.stem<="2024-12")
                    if symbol=="BTCUSDT":
                        p=ROOT/"funding_repairs"/"BTCUSDT.json"
                        if p.exists():sources.append(p)
                        paths=sorted((folder/"predictions").glob("*.parquet"))
                    elif symbol in oldrows:paths=sorted((ROOT/"multi_asset"/oldrows[symbol]["experiment"]/"predictions").glob("*.parquet"))
                    else:paths=[]
                    if symbol=="BTCUSDT" or symbol in oldrows:require(paths,"Frozen baseline cache missing "+symbol)
                    fit_baseline=not paths
                    row["baseline_mode"]="new_fixed_fold_fit" if fit_baseline else "frozen_cache"
                    key={"symbol":symbol,"protocol":SPEC,"source":digest(sources),"baseline_source":("new_fixed_fold_fit" if fit_baseline else digest(paths)),
                         "code":code,"numpy":np.__version__,"polars":pl.__version__,"sklearn":sklearn.__version__}
                    row["identity"]=identity(key)
                    funding=load_funding(symbol)
                    row["unknown_funding_events"]=int(funding[3][-1])
                    row["unknown_funding_event_times"]=funding[1][np.diff(funding[3])>0].tolist()
                    cache_id=identity({"raw":digest(raw),"builder":digest([Path("research_engine.py"),Path("feature_engine.py"),Path("data_engine.py")]),"polars":pl.__version__,"numpy":np.__version__,"TFS":TFS,"features":COLS,"horizons":HORIZONS})
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
            for name,signals in [("fixed_binary",baseline),("nested_joint_search",all_signals)]:
                for fee,slip in COSTS:
                    row={"strategy":name,"fee_bps":fee,"slippage_bps":slip};out["portfolios"].append(row)
                    try:wallet(signals,funds,first,last,fee,slip,row)
                    except Exception:row.update(status="blocked",error=traceback.format_exc())
                    print(name,fee,slip,row["status"],flush=True)
            out["status"]="pass" if all(r["status"]=="pass" for r in out["assets"]+out["portfolios"]) else "partial"
        except Exception:out.update(status="fail",error=traceback.format_exc())
        out["elapsed_seconds"]=time.monotonic()-started
        tmp=report.with_suffix(".tmp")
        with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:z.writestr("metrics.json",json.dumps(out,allow_nan=False))
        tmp.replace(report);print(report,out["status"],flush=True)
    return 0 if out["status"]=="pass" else 1

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--symbols",nargs="+",choices=CRYPTO_SYMBOLS,default=ASSETS)
    a=p.parse_args()
    raise SystemExit(run(list(dict.fromkeys(a.symbols))))
