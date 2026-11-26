"""Multi-asset data readiness gate; no training, no downloads, development only."""
import os
os.environ.setdefault("POLARS_MAX_THREADS","1")
import json,zipfile
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import polars as pl
from universe import CRYPTO_SYMBOLS
from robustness_engine import A,B
ROOT=Path("data")
MIN=60000
def inspect(symbol):
    raw=[p for p in sorted((ROOT/"futures"/symbol/"1m").glob("*.parquet")) if "2021-01"<=p.stem<="2024-12"]
    r={"symbol":symbol,"months":len(raw),"rows":0,"gap_minutes":0,"invalid_rows":0,"first":None,"last":None,"funding_months":0,"mark_months":0}
    prev=None
    for p in raw:
        x=pl.read_parquet(p,columns=["open_time","open","high","low","close","volume"])
        x=x.filter(pl.col("open_time").is_between(A,B,closed="left")).sort("open_time")
        if not x.height:continue
        t=x["open_time"].to_numpy()
        if r["first"] is None:r["first"]=int(t[0])
        if prev is not None:r["gap_minutes"]+=max(0,int((t[0]-prev)//MIN-1))
        d=np.diff(t)
        r["gap_minutes"]+=int(np.maximum(d//MIN-1,0).sum())
        r["invalid_rows"]+=int(np.sum((d<=0)|(d%MIN!=0)))+int(np.sum(t%MIN!=0))
        r["invalid_rows"]+=x.filter(~pl.all_horizontal([pl.col(k).is_finite().fill_null(False)&(pl.col(k)>0) for k in ["open","high","low","close"]])|~(pl.col("volume").is_finite().fill_null(False)&(pl.col("volume")>=0))|(pl.col("high")<pl.max_horizontal("open","close","low"))|(pl.col("low")>pl.min_horizontal("open","close","high"))).height
        r["rows"]+=x.height;prev=int(t[-1]);r["last"]=prev
    for kind in ["funding","mark"]:
        r[kind+"_months"]=sum("2021-01"<=p.stem<="2024-12" for p in (ROOT/kind/symbol).glob("*.parquet"))
    r["full_window"]=r["first"]==A and r["last"]==B-MIN
    r["candidate_ready"]=r["full_window"] and r["gap_minutes"]==0 and r["invalid_rows"]==0 and r["funding_months"]==48 and r["mark_months"]==48
    return r
def main():
    out={"scope":"Development-only raw minute coverage; file-presence checks for funding/mark only, not economic eligibility or listing verification; fixed present-day universe has selection bias","results":[]}
    for symbol in CRYPTO_SYMBOLS:
        try:r=inspect(symbol)
        except Exception as e:r={"symbol":symbol,"error":str(e),"candidate_ready":False}
        out["results"].append(r);print(symbol,r.get("candidate_ready"),r.get("gap_minutes"),flush=True)
    p=ROOT/"reports"/"24_multi_asset_readiness.zip";p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(".tmp")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:z.writestr("metrics.json",json.dumps(out,allow_nan=False))
    tmp.replace(p);print(p)
if __name__=="__main__":main()
