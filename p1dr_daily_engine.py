"""Cross-sectional pilot: fixed 10-asset balanced panel, no retraining.
Ranking diagnostics and fee/slippage-only long-short portfolios.
"""
import os
os.environ.setdefault("POLARS_MAX_THREADS","1")
import json,zipfile,traceback
from pathlib import Path
import numpy as np
import polars as pl
from baseline_engine import digest
from robustness_engine import A,B,DAY,ci
ROOT=Path("data")
H=3600000
def audit(out):
    with zipfile.ZipFile(ROOT/"reports"/"25_multi_asset_transfer.zip") as z:r=json.loads(z.read("metrics.json"))
    assert r["status"]=="pass" and len(r["results"])==10
    frames=[];sources=[]
    folder,=list((ROOT/"experiments").glob("6823506d*"))
    man=json.loads((folder/"manifest.json").read_text())
    MIN=60000
    for row in r["results"]:
        s=row["symbol"]
        raw=sorted(p for p in (ROOT/"futures"/s/"1m").glob("*.parquet") if "2021-01"<=p.stem<="2024-12")
        assert len(raw)==48
        closes=[];opens=[]
        for p in raw:
            x=pl.read_parquet(p,columns=["open_time","open","close"]).sort("open_time")
            assert np.all(np.diff(x["open_time"].to_numpy())==MIN)
            closes.append(x.filter(pl.col("open_time")%DAY==DAY-MIN).select((pl.col("open_time")+MIN).alias("decision_time"),"close"))
            opens.append(x.filter(pl.col("open_time")%DAY==15*MIN).select("open_time","open"))
        f=pl.concat(closes).sort("decision_time")
        assert np.array_equal(f["decision_time"].to_numpy(),np.arange(A+DAY,B+1,DAY))
        p=pl.concat(opens).sort("open_time")
        assert np.array_equal(p["open_time"].to_numpy(),np.arange(A+15*MIN,B,DAY))
        f=f.with_columns(pl.lit(s).alias("symbol"),(pl.col("close")/pl.col("close").shift(1)-1).alias("mom24"),(pl.col("decision_time")+15*MIN).alias("entry_time"),(pl.col("decision_time")+15*MIN+DAY).alias("label_end"))
        f=f.join(p.rename({"open_time":"entry_time","open":"entry_price"}),on="entry_time",validate="1:1").join(p.rename({"open_time":"label_end","open":"exit_price"}),on="label_end",validate="1:1")
        f=f.filter(pl.col("mom24").is_finite()&(pl.col("entry_price")>0)&(pl.col("exit_price")>0)).with_columns((pl.col("exit_price")/pl.col("entry_price")-1).alias("target_return"))
        f=f.filter(pl.any_horizontal([(pl.col("decision_time")>=fd["test_start"])&(pl.col("label_end")<fd["test_end"]) for fd in man["folds"]]))
        frames.append(f);sources+=raw
    panel=pl.concat(frames)
    panel=panel.filter(pl.len().over("decision_time")==10).sort("decision_time","symbol")
    times=panel["decision_time"].unique().sort().to_numpy()
    assert len(times)>0 and panel.height==len(times)*10
    # Fixed UTC daily schedule; avoids overlapping baskets.
    scheduled=panel.filter(pl.col("decision_time")%DAY==0)
    out.update({"source_hash":digest(sources+[Path(__file__)]),"balanced_hours":len(times),"scheduled_hours":scheduled.height//10,"symbols":sorted(panel["symbol"].unique().to_list()),"results":[]})
    for name,sign,col in [("momentum24",1,"mom24"),("reversal24",-1,"mom24")]:
        p=panel.with_columns((sign*pl.col(col)).alias("score"))
        # Spearman IC with average ranks for ties.
        ic=p.with_columns(pl.col("score").rank("average").over("decision_time").alias("a"),pl.col("target_return").rank("average").over("decision_time").alias("b")).group_by("decision_time").agg(pl.corr("a","b").alias("ic")).sort("decision_time")
        v=ic["ic"].to_numpy();v=v[np.isfinite(v)]
        q=p.filter(pl.col("decision_time")%DAY==0).sort(["decision_time","score","symbol"])
        # Bottom 2 short, top 2 long; 25% gross notional each.
        q=q.with_columns(pl.int_range(pl.len()).over("decision_time").alias("rank")).filter((pl.col("rank")<2)|(pl.col("rank")>=8)).with_columns(pl.when(pl.col("rank")<2).then(-1).otherwise(1).alias("side"))
        rows=[]
        for fee,slip in [(0,0),(5,1),(10,5)]:
            s=pl.col("side");a=pl.col("entry_price")*(1+s*slip/10000);b=pl.col("exit_price")*(1-s*slip/10000)
            basket=q.with_columns((s*(b-a)/a-fee/10000*(a+b)/a).alias("leg")).group_by("decision_time","label_end").agg((pl.col("leg").sum()/4).alias("r"),pl.len().alias("legs")).sort("decision_time")
            assert (basket["legs"]==4).all()
            ret=basket["r"].to_numpy();eq=np.r_[0,np.cumsum(ret)]
            first=int(times.min())//DAY*DAY;last=(int(basket["label_end"].max())//DAY+1)*DAY
            ix=(basket["label_end"].to_numpy()-first)//DAY
            daily=np.bincount(ix,weights=ret,minlength=(last-first)//DAY);counts=np.bincount(ix,minlength=len(daily))
            loss=-ret[ret<0].sum()
            rows.append({"fee_bps":fee,"slippage_bps":slip,"baskets":len(ret),"expectancy":float(ret.mean()),"pnl":float(ret.sum()),"profit_factor":float(ret[ret>0].sum()/loss) if loss else None,"fixed_notional_drawdown":float(max(np.maximum.accumulate(eq)-eq)),"bootstrap":[ci(daily,counts,b) for b in (7,30)]})
        out["results"].append({"name":name,"mean_spearman_ic":float(v.mean()) if len(v) else None,"ic_hours":len(v),"fee_slip_only":rows})
    out["status"]="pass"
def main():
    out={"scope":"Liu P1DR-inspired daily ranking; ten fixed survivor assets, 24h past simple return, 24h forward holding, 15m delay, original purged OOS calendar, equal gross notional long/short, full round trip, no funding, no tuning or holdout; not paper replication"}
    try:audit(out)
    except Exception:out.update(status="fail",error=traceback.format_exc())
    p=ROOT/"reports"/"31_p1dr_daily.zip";p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(".tmp")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:z.writestr("metrics.json",json.dumps(out,allow_nan=False))
    tmp.replace(p);print(p,out["status"])
if __name__=="__main__":main()
