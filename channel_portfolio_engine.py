"""Channel 100 portfolio audit, bounded monthly mark scans, no training."""
import os
os.environ.setdefault("POLARS_MAX_THREADS","1")
import json,zipfile,traceback,hashlib
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import polars as pl
from portfolio_engine import require,ROOT,MIN
from robustness_engine import A,B,DAY,ledger
from funding_repair_engine import apply
from execution_engine import score_trades

def main_audit(out):
    path=ROOT/"reports"/"14_channel_audit.zip"
    with zipfile.ZipFile(path) as z:r=json.loads(z.read("metrics.json"))
    require(r["status"]=="pass" and not r["affected_funding_trades"],"Channel audit not passed")
    t=pl.DataFrame(r["trade_ledger"]).sort("entry_time")
    require(t.height==127 and t["entry_time"].min()>=A and t["label_end"].max()<B,"Wrong trade scope")
    e=t["entry_time"].to_numpy();z=t["label_end"].to_numpy()
    require(np.all(e<z) and np.all(e[1:]>=z[:-1]),"Overlapping trades")
    folder,=list((ROOT/"experiments").glob("6823506d*"))
    m=json.loads((folder/"manifest.json").read_text())
    first=m["folds"][0]["test_start"];last=m["folds"][-1]["test_end"]
    require(first<=e[0] and z[-1]<last,"Invalid evaluation window")
    fund=apply(ledger(ROOT)).sort("event_time")
    missing=fund.filter(pl.col("mark_price").is_null())["event_time"].to_numpy()
    require(not np.any(np.searchsorted(missing,z,side="right")>np.searchsorted(missing,e,side="right")),"Unknown funding")
    fund=fund.filter(pl.col("mark_price").is_not_null())
    # Allocate dense arrays only for the active interval (~1.8M minutes).
    grid=np.arange(e[0],z[-1],MIN);px=np.full(len(grid),np.nan)
    active=np.zeros(len(grid),dtype=bool)
    for a,b in zip(e,z):active[(a-e[0])//MIN:(b-e[0])//MIN]=True
    sources={}
    for p in sorted((ROOT/"mark"/"BTCUSDT").glob("*.parquet")):
        if not datetime.fromtimestamp(e[0]/1000,timezone.utc).strftime("%Y-%m")<=p.stem<=datetime.fromtimestamp((z[-1]-1)/1000,timezone.utc).strftime("%Y-%m"):continue
        x=pl.scan_parquet(p).filter(pl.col("event_time").is_between(int(e[0]),int(z[-1]),closed="left")).select("event_time","open").collect()
        overlay=ROOT/"mark_repairs"/"BTCUSDT"/p.name
        if overlay.exists():
            meta=json.loads(overlay.with_suffix(".json").read_text())
            require(hashlib.sha256(overlay.read_bytes()).hexdigest()==meta["overlay_sha256"],"Overlay hash mismatch")
            y=pl.read_parquet(overlay).filter(pl.col("event_time").is_between(int(e[0]),int(z[-1]),closed="left")).cast(x.schema)
            x=pl.concat([x,y])
            sources[str(overlay)]=meta["overlay_sha256"]
        ts=x["event_time"].to_numpy();v=x["open"].to_numpy()
        require(np.all(ts%MIN==0) and len(np.unique(ts))==len(ts),"Invalid mark timestamps")
        ix=(ts-e[0])//MIN
        require(np.all(np.isnan(px[ix])),"Cross-partition duplicate marks")
        px[ix]=v
        sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    # Daily recovery overlay may fill missing or invalid marks only.
    for p in sorted((ROOT/"channel_mark_repairs"/"BTCUSDT").glob("*.parquet")):
        meta=json.loads(p.with_suffix(".json").read_text())
        sha=hashlib.sha256(p.read_bytes()).hexdigest()
        require(sha==meta["overlay_sha256"],"Channel repair hash mismatch")
        x=pl.read_parquet(p).filter(pl.col("event_time").is_between(int(e[0]),int(z[-1]),closed="left"))
        ts=x["event_time"].to_numpy();v=x["open"].to_numpy()
        require(np.all(ts%MIN==0) and len(np.unique(ts))==len(ts) and np.all(np.isfinite(v)&(v>0)),"Invalid recovered marks")
        ix=(ts-e[0])//MIN
        known=np.isfinite(px[ix])&(px[ix]>0)
        require(np.all(px[ix[known]]==v[known]),"Recovery conflicts with existing mark")
        px[ix[~known]]=v[~known]
        sources[str(p)]=sha
    bad=active&(~np.isfinite(px)|(px<=0))
    holes=grid[bad]
    out.update({"audit_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"mark_sources":sources,"expected_minutes":int(active.sum()),"missing_or_invalid_minutes":len(holes),"gap_ranges":[],"scenarios":[]})
    if len(holes):
        for a in np.split(holes,np.where(np.diff(holes)!=MIN)[0]+1):
            out["gap_ranges"].append([int(a[0]),int(a[-1]+MIN)])
        out["status"]="mark_repair_required";return
    for fee,slip in [(5,1),(10,5)]:
        scored=score_trades(t,fee,slip,fund,{"BTCUSDT":[(A,B)]}).sort("entry_time")
        expected=next(v["pnl"] for v in r["metrics"] if v["fee_bps"]==fee and v["slippage_bps"]==slip)
        require(np.isclose(scored["net_return"].sum(),expected,atol=1e-10),"Fixed notional reconciliation failed")
        for frac in (.25,.5,1.):
            equity=peak=1.;dd=0.;daily={};min_equity=1.;status="pass"
            for tr in scored.iter_rows(named=True):
                a,b=tr["entry_time"],tr["label_end"];side=tr["side"]
                times=grid[(a-e[0])//MIN:(b-e[0])//MIN]
                v=px[(a-e[0])//MIN:(b-e[0])//MIN]
                start=equity;q=start*frac/tr["entry_fill"];cost=fee/10000
                ff=fund.filter((pl.col("event_time")>a)&(pl.col("event_time")<=b))
                fc=np.r_[0.,np.cumsum(ff["funding_rate"].to_numpy()*ff["mark_price"].to_numpy()*side*q)]
                floating=start-q*tr["entry_fill"]*cost+side*q*(v-tr["entry_fill"])-fc[np.searchsorted(ff["event_time"].to_numpy(),times,side="right")]
                equity=start*(1+frac*tr["net_return"])
                require(np.isclose(equity,start+side*q*(tr["exit_fill"]-tr["entry_fill"])-cost*q*(tr["entry_fill"]+tr["exit_fill"])-fc[-1]),"Cash reconciliation")
                vals=np.r_[floating,equity];tt=np.r_[times,b]
                min_equity=min(min_equity,float(vals.min()))
                if min_equity<=0:status="nonpositive_equity";break
                peaks=np.maximum.accumulate(np.r_[peak,vals])[1:];peak=float(peaks[-1]);dd=max(dd,float(np.max(1-vals/peaks)))
                ix=np.r_[np.where(np.diff(tt//DAY)!=0)[0],len(tt)-1]
                for i in ix:daily[int(tt[i])//DAY]=float(vals[i])
            row={"fee_bps":fee,"slippage_bps":slip,"fraction":frac,"status":status}
            if status=="pass":
                nav=[];prev=1.
                for day in range(first//DAY,(last-1)//DAY+1):
                    prev=daily.get(day,prev);nav.append(prev)
                ret=np.diff(np.r_[1.,nav])/np.r_[1.,nav][:-1]
                row.update({"return":equity-1,"max_mtm_drawdown":dd,"daily_sharpe_zero_rf":float(ret.mean()/ret.std(ddof=1)*np.sqrt(365)) if ret.std(ddof=1)>0 else None,"daily_equity":nav,"exposure":float(np.sum(z-e)/(last-first))})
            out["scenarios"].append(row);print(fee,slip,frac,status,flush=True)
    out["status"]="pass"
def main():
    out={"scope":"Frozen channel audit ledger, common incumbent manifest calendar; minute-open mark proxy; no liquidation model or intraminute extremes; fixed fractions not optimized"}
    try:main_audit(out)
    except Exception:out.update(status="fail",error=traceback.format_exc())
    p=ROOT/"reports"/"15_channel_portfolio.zip";tmp=p.with_suffix(".tmp")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:z.writestr("metrics.json",json.dumps(out,allow_nan=False))
    tmp.replace(p);print(p,out["status"])
if __name__=="__main__":main()
