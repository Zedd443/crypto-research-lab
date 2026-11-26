"""Channel 100 independent hourly execution audit; no training or holdout."""
import os
os.environ.setdefault("POLARS_MAX_THREADS","1")
import json,zipfile,traceback
from pathlib import Path
import numpy as np
import polars as pl
from robustness_engine import A,B,DAY,ledger,metrics
from execution_engine import score_trades
from baseline_engine import digest
ROOT=Path("data")
H=3600000
def audit(out):
    folder,=list((ROOT/"experiments").glob("6823506d*"))
    paths=sorted((folder/"predictions").glob("*.parquet"))
    f=pl.read_parquet(paths).sort("decision_time")
    first=int(f["decision_time"].min()); last=int(f["decision_time"].max())
    assert A<=first<last<B and last+900000<B
    src=[p for p in sorted((ROOT/"resampled"/"BTCUSDT"/"1h").glob("*.parquet")) if "2020-12"<=p.stem<="2024-12"]
    x=pl.read_parquet(src).sort("open_time")
    assert x["open_time"].n_unique()==x.height
    assert (x["close_time"]==x["open_time"]+H-1).all()
    x=x.with_columns((pl.col("open_time").diff().fill_null(H)!=H).cast(pl.Int64).cum_sum().alias("seg"),(pl.col("close_time")+1).alias("decision_time"))
    hi=pl.col("high").shift(1).rolling_max(100).over("seg")
    lo=pl.col("low").shift(1).rolling_min(100).over("seg")
    x=x.with_columns(hi.alias("hi"),lo.alias("lo")).filter(pl.col("decision_time").is_between(first,last))
    grid=np.arange(first,last+H,H)
    out["missing_hourly_decisions"]=np.setdiff1d(grid,x["decision_time"].to_numpy()).tolist()
    out["warmup_missing"]=x.filter(pl.col("hi").is_null()|pl.col("lo").is_null())["decision_time"].to_list()
    if out["missing_hourly_decisions"] or out["warmup_missing"]:raise ValueError("Incomplete independent hourly signal grid")
    x=x.with_columns(pl.when(pl.col("close")>pl.col("hi")).then(1).when(pl.col("close")<pl.col("lo")).then(-1).otherwise(0).alias("side"))
    pos=0; start=None; records=[]
    for d,s in x.select("decision_time","side").iter_rows():
        desired=0 if d==last else (s or pos)
        if desired==pos:continue
        if pos:records.append({"symbol":"BTCUSDT","decision_time":start,"entry_time":start+900000,"label_end":d+900000,"side":pos})
        pos=desired;start=d if pos else None
    t=pl.DataFrame(records)
    wanted=sorted(set(t["entry_time"].to_list()+t["label_end"].to_list()))
    from datetime import datetime,timezone
    months=sorted({datetime.fromtimestamp(v/1000,timezone.utc).strftime("%Y-%m") for v in wanted})
    raw=[ROOT/"futures"/"BTCUSDT"/"1m"/(m+".parquet") for m in months]
    prices=pl.concat([pl.scan_parquet(p).filter(pl.col("open_time").is_in(wanted)).select("open_time","open").collect() for p in raw])
    out["missing_execution_minutes"]=np.setdiff1d(wanted,prices["open_time"].to_numpy()).tolist()
    if out["missing_execution_minutes"] or prices["open_time"].n_unique()!=prices.height:raise ValueError("Invalid execution coverage")
    t=t.join(prices.rename({"open_time":"entry_time","open":"entry_price"}),on="entry_time",validate="m:1").join(prices.rename({"open_time":"label_end","open":"exit_price"}),on="label_end",validate="m:1").sort("entry_time")
    fund=ledger(ROOT)
    missing=fund.filter(pl.col("mark_price").is_null())["event_time"].to_numpy()
    affected=[]
    for row in t.iter_rows(named=True):
        holes=missing[(missing>row["entry_time"])&(missing<=row["label_end"])]
        if len(holes):affected.append({"entry_time":row["entry_time"],"label_end":row["label_end"],"missing_funding_times":holes.tolist()})
    out.update({"trade_count":t.height,"affected_funding_trades":affected,"trade_ledger":t.to_dicts(),"source_hash":digest(src+paths+raw+[Path(__file__)]),"exposure":float((t["label_end"]-t["entry_time"]).sum()/(last+900000-first))})
    if affected:
        out["status"]="funding_repair_required"
        return
    out["metrics"]=[{"fee_bps":fee,"slippage_bps":slip,**metrics(score_trades(t,fee,slip,fund,{"BTCUSDT":[(A,B)]}),first//DAY*DAY,((last+900000)//DAY+1)*DAY)} for fee,slip in [(5,1),(10,5)]]
    out["status"]="pass"
def main():
    out={"scope":"Channel 100, frozen 15m delay, same start/end decision dates as incumbent but independent hourly eligibility, opposite breakout reversal, terminal close, no missing funding exclusion; fixed-notional diagnostic only"}
    try:audit(out)
    except Exception:out.update(status="fail",error=traceback.format_exc())
    p=ROOT/"reports"/"14_channel_audit.zip";p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(".tmp")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:z.writestr("metrics.json",json.dumps(out,allow_nan=False))
    tmp.replace(p);print(p,out["status"])
if __name__=="__main__":main()
