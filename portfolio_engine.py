"""Single-position BTC development portfolio replay, no training or holdout."""
import os
os.environ.setdefault("POLARS_MAX_THREADS","1")
import json, zipfile, traceback, hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import polars as pl
from robustness_engine import ledger,pick,A,B,DAY
from execution_engine import score_trades
from experiment_store import identity

ROOT=Path("data")
PREFIX="6823506d"
MIN=60000

def require(ok,msg):
    if not ok:raise ValueError(msg)

def replay(trades,marks,fraction,first,last,fee):
    equity=1.; peak=1.; dd=0.; equity_daily={}; curve=[]; records=[]
    for t in trades.iter_rows(named=True):
        e=int(t["entry_time"]); z=int(t["label_end"]); side=t["side"]
        month=datetime.fromtimestamp(e/1000,timezone.utc).strftime("%Y-%m")
        endmonth=datetime.fromtimestamp(z/1000,timezone.utc).strftime("%Y-%m")
        paths=[ROOT/"mark"/"BTCUSDT"/(m+".parquet") for m in sorted({month,endmonth})]
        x=pl.scan_parquet(paths).filter(pl.col("event_time").is_between(e,z,closed="left")).select("event_time","open").collect().sort("event_time")
        ts=x["event_time"].to_numpy(); px=x["open"].to_numpy()
        require(np.array_equal(ts,np.arange(e,z,MIN)),"Missing intratrade mark minute")
        require(np.all(np.isfinite(px)&(px>0)),"Invalid mark price")
        start=equity; notional=start*fraction; q=notional/t["entry_fill"]
        f=marks.filter((pl.col("event_time")>e)&(pl.col("event_time")<=z))
        ft=f["event_time"].to_numpy(); fc=np.r_[0.,np.cumsum(f["funding_rate"].to_numpy()*f["mark_price"].to_numpy()*q*side)]
        floating=start-q*t["entry_fill"]*fee+side*q*(px-t["entry_fill"])-fc[np.searchsorted(ft,ts,side="right")]
        equity=start+notional*t["net_return"]
        require(np.isclose(equity,start+side*q*(t["exit_fill"]-t["entry_fill"])-fee*q*(t["entry_fill"]+t["exit_fill"])-fc[-1]),"Cash reconciliation")
        times=np.r_[ts,z]; vals=np.r_[floating,equity]
        require(np.all(vals>0),"Nonpositive equity: margin model required")
        peaks=np.maximum.accumulate(np.r_[peak,vals])[1:]
        dd=max(dd,float(np.max(1-vals/peaks)));peak=float(peaks[-1])
        for ti,v in zip(times,vals):
            equity_daily[int(ti)//DAY]=float(v)
        curve.extend(zip(times.tolist(),vals.tolist()))
        records.append({**t,"equity_before":start,"equity_after":equity,"notional":notional,"quantity":q})
    daily=[]; prev=1.
    for day in range(first//DAY,(last-1)//DAY+1):
        prev=equity_daily.get(day,prev);daily.append(prev)
    r=np.diff(np.r_[1.,daily])/np.r_[1.,daily][:-1]
    exposure=sum(t["label_end"]-t["entry_time"] for t in records)/(last-first)
    return {"fraction":fraction,"return":equity-1,"max_mtm_drawdown":dd,"exposure":exposure,"daily_sharpe_zero_rf":float(np.mean(r)/np.std(r,ddof=1)*np.sqrt(365)) if np.std(r,ddof=1)>0 else None,"trades":len(records),"daily_equity":daily,"ledger":records,"minute_equity":curve}

def audit(out):
    matches=list((ROOT/"experiments").glob(PREFIX+"*"))
    require(len(matches)==1,"Incumbent experiment not uniquely found")
    folder=matches[0];s=json.loads((folder/"spec.json").read_text());m=json.loads((folder/"manifest.json").read_text());r=json.loads((folder/"result.json").read_text())
    require(identity(s)==folder.name and s["manifest_id"]==m["manifest_id"],"Identity mismatch")
    require(s["model"]=="logistic" and s["delay_minutes"]==15 and s["horizon"]==4 and len(s["features"])==11,"Wrong incumbent")
    paths=sorted((folder/"predictions").glob("*.parquet"))
    require({p.stem for p in paths}=={str(f["fold_id"]) for f in m["folds"]},"Incomplete folds")
    f=pl.read_parquet(paths).sort("decision_time")
    require(f["decision_time"].n_unique()==f.height and f["symbol"].unique().to_list()==["BTCUSDT"] and f["decision_time"].min()>=A and f["label_end"].max()<B,"Invalid prediction scope")
    first=m["folds"][0]["test_start"];last=m["folds"][-1]["test_end"]
    fund=ledger(ROOT);missing=fund.filter(pl.col("mark_price").is_null())["event_time"].to_numpy();fund=fund.filter(pl.col("mark_price").is_not_null())
    t=pick(f,"logistic",.7)
    require(not np.any(np.searchsorted(missing,t["label_end"].to_numpy(),side="right")>np.searchsorted(missing,t["entry_time"].to_numpy(),side="right")),"Unknown funding in incumbent; do not silently exclude")
    out.update({"experiment":folder.name,"first":first,"last":last,"prediction_hashes":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},"cash_benchmark_return":0.,"scenarios":[]})
    for fee,slip in [(5,1),(5,3),(10,5)]:
        scored=score_trades(t,fee,slip,fund,{"BTCUSDT":[(A,B)]})
        require(scored.height==104,"Incumbent trade count mismatch")
        if (fee,slip)==(5,1):require(np.isclose(scored["net_return"].sum(),.573242,atol=1e-5),"Incumbent P&L mismatch")
        for frac in (.25,.5,1.):
            result=replay(scored,fund,frac,first,last,fee/10000)
            out["scenarios"].append({"fee_bps":fee,"slippage_bps":slip,"fixed_notional_pnl":float(scored["net_return"].sum()),**result})
            print(fee,slip,frac,"done",flush=True)

def main():
    out={"scope":"Development diagnostic, fixed fractions not optimized; minute-open marks, actual assumed funding boundary (entry,exit], no liquidation or margin tiers; no intra-minute extrema or executable mark fills."}
    try:audit(out);out["status"]="pass"
    except Exception:out["status"]="fail";out["error"]=traceback.format_exc()
    p=ROOT/"reports"/"10_portfolio.zip";p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(".tmp")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:z.writestr("metrics.json",json.dumps(out,allow_nan=False))
    tmp.replace(p);print(p,out["status"])
    if out["status"]!="pass":raise SystemExit(out["error"])
if __name__=="__main__":main()
