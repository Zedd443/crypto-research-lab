"""Funding-aware rescore of frozen multi-asset predictions; no model fits."""
import os
os.environ.setdefault("POLARS_MAX_THREADS","1")
import json,zipfile,traceback
from pathlib import Path
import numpy as np
import polars as pl
from baseline_engine import digest
from robustness_engine import A,B,DAY,pick,metrics
from execution_engine import score_trades
ROOT=Path("data")
H=3600000
MIN=60000
SYMBOLS=["BNBUSDT","ETHUSDT","ATOMUSDT","AVAXUSDT","UNIUSDT"]
def audit_symbol(r,first,last,out):
    symbol=r["symbol"]
    paths=sorted((ROOT/"multi_asset"/r["experiment"]/"predictions").glob("*.parquet"))
    assert len(paths)==42
    f=pl.read_parquet(paths).sort("decision_time")
    assert f["symbol"].unique().to_list()==[symbol] and f["decision_time"].n_unique()==f.height
    assert f["decision_time"].min()>=first and f["label_end"].max()<last
    assert f.height==r["oos_rows"]
    t=pick(f,"logistic",.7)
    assert t.height==r["fee_slip_only"][0]["trades"]
    fp=sorted(p for p in (ROOT/"funding"/symbol).glob("*.parquet") if "2021-01"<=p.stem<="2024-12")
    assert len(fp)==48
    parts=[];sources=list(paths)+fp
    for p in fp:
        x=pl.read_parquet(p).select("event_time","funding_rate","funding_interval_hours")
        mp=ROOT/"mark"/symbol/p.name
        sources.append(mp)
        m=pl.read_parquet(mp,columns=["event_time","open"]).rename({"event_time":"minute","open":"mark_price"})
        parts.append(x.with_columns((pl.col("event_time")//MIN*MIN).alias("minute")).join(m,on="minute",how="left",validate="m:1"))
    fund=pl.concat(parts).filter(pl.col("event_time").is_between(A,B,closed="left")).sort("event_time")
    ts=fund["event_time"].to_numpy()
    interval=fund["funding_interval_hours"].cast(pl.Float64).to_numpy()
    # Strict 8h grid first; nonstandard schedules are reported, not guessed.
    expected=np.arange(A,B,8*H)
    grid=((ts+MIN//2)//(8*H))*(8*H)
    out.update({"funding_events":len(ts),"interval_counts":fund.group_by("funding_interval_hours").len().to_dicts(),"missing_8h_grid":np.setdiff1d(expected,grid).tolist(),"unexpected_8h_grid":np.setdiff1d(grid,expected).tolist(),"duplicate_grid":int(len(grid)-len(np.unique(grid))),"off_grid_events":ts[np.abs(ts-grid)>MIN].tolist()})
    valid_schedule=np.array_equal(grid,expected) and np.all(np.abs(ts-grid)<=MIN) and np.all(interval==8)
    if not valid_schedule:
        out["status"]="schedule_review_required";return
    assert fund["funding_rate"].is_finite().fill_null(False).all()
    valid=pl.col("mark_price").is_finite().fill_null(False)&(pl.col("mark_price")>0).fill_null(False)
    missing=fund.filter(~valid)["event_time"].to_numpy()
    affected=[]
    for tr in t.iter_rows(named=True):
        holes=missing[(missing>tr["entry_time"])&(missing<=tr["label_end"])]
        if len(holes):affected.append({"entry_time":tr["entry_time"],"label_end":tr["label_end"],"missing_funding_times":holes.tolist()})
    out.update({"missing_marks":missing.tolist(),"affected_trades":affected,"trade_count":t.height,"source_hash":digest(sources+[Path(__file__),Path("execution_engine.py")])})
    if affected:
        out["status"]="funding_repair_required";return
    fund=fund.filter(valid).with_columns(pl.lit(symbol).alias("symbol")).select("symbol","event_time","funding_rate","mark_price")
    out["economics"]=[]
    for fee,slip in [(5,1),(10,5)]:
        scored=score_trades(t,fee,slip,fund,{symbol:[(A,B)]})
        out["economics"].append({"fee_bps":fee,"slippage_bps":slip,**metrics(scored,first,last)})
    out["status"]="pass"
def main():
    out={"scope":"Five positive-primary-cost altcoins, fixed predictions, strict standard 8h funding grid gate, minute-open mark proxy; no dropped trades or retraining; nonstandard schedules require separate verification; exploratory fixed-notional results","results":[]}
    try:
        with zipfile.ZipFile(ROOT/"reports"/"25_multi_asset_transfer.zip") as z:report=json.loads(z.read("metrics.json"))
        assert report["status"]=="pass"
        folder,=list((ROOT/"experiments").glob("6823506d*"))
        m=json.loads((folder/"manifest.json").read_text())
        first=m["folds"][0]["test_start"];last=m["folds"][-1]["test_end"]
        for symbol in SYMBOLS:
            row={"symbol":symbol}
            try:audit_symbol(next(r for r in report["results"] if r["symbol"]==symbol),first,last,row)
            except Exception:row.update(status="fail",error=traceback.format_exc())
            out["results"].append(row);print(symbol,row["status"],flush=True)
        out["status"]="pass" if all(r["status"]=="pass" for r in out["results"]) else "partial"
    except Exception:out.update(status="fail",error=traceback.format_exc())
    p=ROOT/"reports"/"26_multi_asset_funding.zip";p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(".tmp")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:z.writestr("metrics.json",json.dumps(out,allow_nan=False))
    tmp.replace(p);print(p,out["status"])
if __name__=="__main__":main()
