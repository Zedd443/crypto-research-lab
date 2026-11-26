"""ProSP data readiness, no training. Inspect exact completed 4h/8h spot endpoints."""
import os
os.environ.setdefault("POLARS_MAX_THREADS","1")
import json,zipfile,traceback
from pathlib import Path
import numpy as np
import polars as pl
from robustness_engine import A,B
from baseline_engine import digest
ROOT=Path("data");H=3600000;MIN=60000
def audit(out):
    with zipfile.ZipFile(ROOT/"reports"/"25_multi_asset_transfer.zip") as z:r=json.loads(z.read("metrics.json"))
    assert r["status"]=="pass" and len(r["results"])==10
    out["results"]=[]; masks={4:[],8:[]}
    for row in r["results"]:
        s=row["symbol"];result={"symbol":s};out["results"].append(result)
        try:
            raw=sorted(p for p in (ROOT/"futures"/s/"1m").glob("*.parquet") if "2021-01"<=p.stem<="2024-12")
            spot=sorted(p for p in (ROOT/"spot"/s).glob("*.parquet") if "2021-01"<=p.stem<="2024-12")
            result.update(raw_months=len(raw),spot_months=len(spot))
            assert len(raw)==48
            for hours in (4,8):
                step=hours*H; expected=np.arange(A+step,B+1,step)
                f=[];q=[]
                for p in raw:
                    x=pl.scan_parquet(p).filter(pl.col("open_time").is_between(A,B,closed="left")).select("open_time","close","volume","quote_volume").collect()
                    assert x["open_time"].n_unique()==x.height
                    y=x.with_columns((pl.col("open_time")//step*step+step).alias("decision_time")).group_by("decision_time").agg(pl.len().alias("n"),pl.col("close").sort_by("open_time").last(),pl.col("volume").sum(),pl.col("quote_volume").sum()).filter(pl.col("n")==step//MIN)
                    f.append(y)
                fut=pl.concat(f).sort("decision_time")
                assert np.array_equal(fut["decision_time"].to_numpy(),expected)
                assert fut.select(pl.all_horizontal([pl.col("close").is_finite()&(pl.col("close")>0),pl.col("volume").is_finite()&(pl.col("volume")>=0),pl.col("quote_volume").is_finite()&(pl.col("quote_volume")>=0)]).all()).item()
                for p in spot:
                    q.append(pl.scan_parquet(p).filter((pl.col("event_time")%step==step-MIN)&pl.col("event_time").is_between(A,B,closed="left")).select((pl.col("event_time")+MIN).alias("decision_time"),pl.col("close").alias("spot_close")).collect())
                if q:
                    sp=pl.concat(q);assert sp["decision_time"].n_unique()==sp.height
                    fut=fut.join(sp,on="decision_time",how="left",validate="1:1")
                else:fut=fut.with_columns(pl.lit(None,dtype=pl.Float64).alias("spot_close"))
                valid=(fut["spot_close"].is_finite()&(fut["spot_close"]>0)).fill_null(False).to_numpy()
                # Lagged cumulative returns over K periods need K+1 contiguous valid endpoints.
                # Current spot basis must also be available; conservative complete history.
                k=35*24//hours
                history=np.convolve(valid.astype(np.int64),np.ones(k+2,dtype=np.int64),"valid")==k+2
                ready=np.r_[np.zeros(k+1,dtype=bool),history]
                # Set 1 requires 12 lagged returns, hence 13 prior endpoints plus current.
                m1=np.arange(len(expected))>=13
                masks[hours].append((s,m1,ready))
                result[str(hours)+"h"]={"expected_bars":len(expected),"futures_bars":fut.height,"spot_valid":int(valid.sum()),"spot_missing":int((~valid).sum()),"first_missing":expected[~valid][:10].tolist(),"set1_history_ready":int(m1.sum()),"set2_history_ready":int(ready.sum()),"set3_history_ready":int((m1&ready).sum())}
            result["source_hash"]=digest(raw+spot);result["status"]="pass"
        except Exception:result.update(status="fail",error=traceback.format_exc())
        print(s,result["status"],flush=True)
    out["balanced_coverage"]={}
    for hours,items in masks.items():
        if len(items)==10:
            out["balanced_coverage"][str(hours)+"h"]={"set1":int(np.logical_and.reduce([v[1] for v in items]).sum()),"set3":int(np.logical_and.reduce([v[1]&v[2] for v in items]).sum())}
    out["status"]="pass" if all(r["status"]=="pass" for r in out["results"]) else "partial"
def main():
    out={"scope":"Development only; ten fixed survivor symbols; exact spot minute closes matched to completed 4h/8h futures bars; 35-day contiguous endpoint history, conservative warmup. Quote volume is candidate turnover proxy, not confirmed identical to paper. No feature fits, no holdout or downloads. Coverage counts include end-of-development feature timestamp only, not label eligibility."}
    try:audit(out)
    except Exception:out.update(status="fail",error=traceback.format_exc())
    p=ROOT/"reports"/"29_prosp_readiness.zip";p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(".tmp")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:z.writestr("metrics.json",json.dumps(out,allow_nan=False))
    tmp.replace(p);print(p,out["status"])
if __name__=="__main__":main()
