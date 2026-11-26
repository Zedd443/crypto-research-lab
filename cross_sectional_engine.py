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
    for row in r["results"]:
        s=row["symbol"];key=row["experiment"]
        base=ROOT/"experiments"/key if s=="BTCUSDT" else ROOT/"multi_asset"/key
        ps=sorted((base/"predictions").glob("*.parquet"))
        assert len(ps)==42
        f=pl.read_parquet(ps).sort("decision_time")
        assert f.height==row["oos_rows"] and f["symbol"].unique().to_list()==[s] and f["decision_time"].n_unique()==f.height
        assert f["decision_time"].min()>=A and f["label_end"].max()<B
        assert ((f["entry_time"]-f["decision_time"])==15*60000).all() and ((f["label_end"]-f["entry_time"])==4*H).all()
        if s=="BTCUSDT":
            raw=sorted(p for p in (ROOT/"features"/s/"1h").glob("*.parquet") if "2021-01"<=p.stem<="2024-12")
            x=pl.read_parquet(raw,columns=["symbol","decision_time","ret_16"])
            # 24h momentum from gap-aware completed hourly closes, not labels.
            bars=pl.read_parquet(sorted(p for p in (ROOT/"resampled"/s/"1h").glob("*.parquet") if "2021-01"<=p.stem<="2024-12"),columns=["open_time","close"]).sort("open_time")
            sources+=raw+sorted(p for p in (ROOT/"resampled"/s/"1h").glob("*.parquet") if "2021-01"<=p.stem<="2024-12")
        else:
            raw=sorted(p for p in (ROOT/"futures"/s/"1m").glob("*.parquet") if "2021-01"<=p.stem<="2024-12")
            # Read only completed hourly close rows; sequential monthly scans.
            bars=pl.concat([pl.scan_parquet(p).filter(pl.col("open_time")%H==H-60000).select((pl.col("open_time")-H+60000).alias("open_time"),"close").collect() for p in raw]).sort("open_time")
            sources+=raw
        assert np.array_equal(bars["open_time"].to_numpy(),np.arange(A,B,H))
        x=bars.select((pl.col("open_time")+H).alias("decision_time"),(pl.col("close").log()-pl.col("close").shift(24).log()).alias("mom24"))
        f=f.join(x,on="decision_time",validate="1:1")
        assert f.height==row["oos_rows"] and f["mom24"].is_finite().all()
        frames.append(f.select("symbol","decision_time","entry_time","label_end","entry_price","exit_price","target_return","prediction","mom24"));sources+=ps
    panel=pl.concat(frames)
    panel=panel.filter(pl.len().over("decision_time")==10).sort("decision_time","symbol")
    times=panel["decision_time"].unique().sort().to_numpy()
    assert len(times)>0 and panel.height==len(times)*10
    # Fixed UTC 4-hour schedule; avoids overlapping baskets.
    scheduled=panel.filter(pl.col("decision_time")%(4*H)==0)
    out.update({"source_hash":digest(sources+[Path(__file__)]),"balanced_hours":len(times),"scheduled_hours":scheduled.height//10,"symbols":sorted(panel["symbol"].unique().to_list()),"results":[]})
    for name,sign,col in [("logistic_rank",1,"prediction"),("momentum24",1,"mom24"),("reversal24",-1,"mom24")]:
        p=panel.with_columns((sign*pl.col(col)).alias("score"))
        # Spearman IC with average ranks for ties.
        ic=p.with_columns(pl.col("score").rank("average").over("decision_time").alias("a"),pl.col("target_return").rank("average").over("decision_time").alias("b")).group_by("decision_time").agg(pl.corr("a","b").alias("ic")).sort("decision_time")
        v=ic["ic"].to_numpy();v=v[np.isfinite(v)]
        q=p.filter(pl.col("decision_time")%(4*H)==0).sort(["decision_time","score","symbol"])
        # Bottom 2 short, top 2 long; 25% gross notional each.
        q=q.with_columns(pl.int_range(pl.len()).over("decision_time").alias("rank")).filter((pl.col("rank")<2)|(pl.col("rank")>=8)).with_columns(pl.when(pl.col("rank")<2).then(-1).otherwise(1).alias("side"))
        rows=[]
        for fee,slip in [(5,1),(10,5)]:
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
    out={"scope":"Exploratory fixed survivor panel; ranking existing independent logistic probabilities, 24h momentum and reversal controls; UTC 4h scheduled top/bottom two, gross notional one, equal weight, no funding, no portfolio compounding, no tuning or holdout. Balanced panel conditional on eligibility. Not ProSP replication."}
    try:audit(out)
    except Exception:out.update(status="fail",error=traceback.format_exc())
    p=ROOT/"reports"/"27_cross_sectional.zip";p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(".tmp")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:z.writestr("metrics.json",json.dumps(out,allow_nan=False))
    tmp.replace(p);print(p,out["status"])
if __name__=="__main__":main()
