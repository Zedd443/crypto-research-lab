"""Frozen BTC specification transfer pilot, independent per-asset fits.
Only predictive and fee/slippage-only economics until funding is verified.
"""
import os
os.environ.setdefault("POLARS_MAX_THREADS","1")
import json,zipfile,traceback,hashlib
from pathlib import Path
import numpy as np
import polars as pl
import sklearn
from baseline_engine import digest,fit_score
from feature_engine import technical,BASE_FEATURES
from validation_protocol import Fold,select_fold
from experiment_store import identity
from robustness_engine import A,B,DAY,pick,metrics
from data_engine import aggregate,atomic_write
ROOT=Path("data")
H=3600000
MIN=60000
SYMBOLS=["ETHUSDT","BNBUSDT","DOGEUSDT","ADAUSDT","LINKUSDT","AVAXUSDT","BCHUSDT","UNIUSDT","ATOMUSDT"]
def build(symbol,raw,cache):
    target=cache/"dataset.parquet"
    if target.exists():return pl.read_parquet(target)
    bars=[];points=[]
    for p in raw:
        f=pl.read_parquet(p).filter(pl.col("open_time").is_between(A,B,closed="left")).sort("open_time")
        ts=f["open_time"].to_numpy()
        assert len(ts)>0 and np.all(np.diff(ts)==MIN) and np.all(ts%MIN==0)
        bars.append(aggregate(f,"1h"))
        points.append(f.filter(pl.col("open_time")%H==15*MIN).select("open_time","open"))
    y=pl.concat(bars).sort("open_time")
    ts=y["open_time"].to_numpy()
    assert np.array_equal(ts,np.arange(A,B,H)),"Incomplete hourly grid"
    p=pl.concat(points).sort("open_time")
    assert np.array_equal(p["open_time"].to_numpy(),np.arange(A+15*MIN,B,H)),"Incomplete execution grid"
    x=technical(y,H).with_columns(pl.lit(symbol).alias("symbol"))
    x=x.with_columns((pl.col("decision_time")+15*MIN).alias("entry_time"),(pl.col("decision_time")+15*MIN+4*H).alias("label_end"))
    x=x.join(p.rename({"open_time":"entry_time","open":"entry_price"}),on="entry_time",validate="1:1").join(p.rename({"open_time":"label_end","open":"exit_price"}),on="label_end",validate="1:1")
    x=x.filter(pl.all_horizontal([pl.col(k).is_finite().fill_null(False) for k in BASE_FEATURES])&(pl.col("entry_price")>0)&(pl.col("exit_price")>0)&(pl.col("label_end")<B)).with_columns((pl.col("exit_price")/pl.col("entry_price")).log().alias("target_return"))
    x=x.select("symbol","decision_time","entry_time","label_end","entry_price","exit_price","target_return",*BASE_FEATURES)
    atomic_write(x,target)
    return x
def evaluate(f,first,last):
    p=f["prediction"].to_numpy();y=f["target_return"].to_numpy()>0
    t=pick(f,"logistic",.7)
    results=[]
    # These are NOT funding-adjusted returns; use only as preliminary diagnostics.
    for fee,slip in [(5,1),(10,5)]:
        s=pl.col("side");a=pl.col("entry_price")*(1+s*slip/10000);b=pl.col("exit_price")*(1-s*slip/10000)
        scored=t.with_columns((s*(b-a)/a-fee/10000*(a+b)/a).alias("net_return"),pl.lit(0.).alias("funding_return"))
        m=metrics(scored,first,last);m.pop("funding",None)
        results.append({"fee_bps":fee,"slippage_bps":slip,**m})
    return {"oos_rows":f.height,"brier":float(np.mean((p-y)**2)),"naive_brier":float(np.mean((f["baseline_prediction"].to_numpy()-y)**2)),"DA":float(np.mean((p>=.5)==y)),"fee_slip_only":results}
def audit(out):
    folder,=list((ROOT/"experiments").glob("6823506d*"))
    spec=json.loads((folder/"spec.json").read_text());man=json.loads((folder/"manifest.json").read_text())
    assert identity(spec)==folder.name and spec["manifest_id"]==man["manifest_id"]
    assert spec["model"]=="logistic" and spec["horizon"]==4 and spec["delay_minutes"]==15
    params=json.loads((folder/"result.json").read_text())["params"];cols=spec["features"]
    assert set(cols)==set(BASE_FEATURES)-{"body"}
    code=digest([Path(__file__),Path("baseline_engine.py"),Path("feature_engine.py"),Path("validation_protocol.py"),Path("data_engine.py"),Path("robustness_engine.py")])
    first=man["folds"][0]["test_start"];last=man["folds"][-1]["test_end"]
    out["results"]=[]
    btc=pl.read_parquet(sorted((folder/"predictions").glob("*.parquet"))).sort("decision_time")
    out["results"].append({"symbol":"BTCUSDT","status":"pass","experiment":folder.name,**evaluate(btc,first,last)})
    for symbol in SYMBOLS:
        try:
            raw=[p for p in sorted((ROOT/"futures"/symbol/"1m").glob("*.parquet")) if "2021-01"<=p.stem<="2024-12"]
            assert len(raw)==48
            source=digest(raw)
            key=identity({"source":source,"code":code,"incumbent":folder.name,"symbol":symbol,"params":params,"sklearn":sklearn.__version__,"polars":pl.__version__})
            cache=ROOT/"multi_asset"/key
            f=build(symbol,raw,cache);paths=[]
            for fd in man["folds"]:
                fold=Fold(**fd)
                train,test=select_fold(f,fold)
                assert train.height>=100 and test.height>=30
                p=cache/"predictions"/(str(fold.fold_id)+".parquet")
                if not p.exists():fit_score(train,test,cols,"logistic",params,1,spec["seed"],p)
                v=pl.read_parquet(p)
                keys=["symbol","decision_time","entry_time","label_end","entry_price","exit_price","target_return"]
                assert v.select(keys).equals(test.select(keys))
                paths.append(p)
            oos=pl.read_parquet(paths).sort("decision_time")
            assert oos["decision_time"].n_unique()==oos.height and oos["label_end"].max()<B
            out["results"].append({"symbol":symbol,"status":"pass","experiment":key,"source":source,**evaluate(oos,first,last)})
            print(symbol,"done",flush=True)
        except Exception:out["results"].append({"symbol":symbol,"status":"fail","error":traceback.format_exc()});print(symbol,"failed",flush=True)
    out["status"]="pass" if all(r["status"]=="pass" for r in out["results"]) else "partial"
def main():
    out={"scope":"Independent per-symbol retraining of frozen BTC specification, not zero-shot model transfer; same chronological folds, 15m delay, 4h horizon, .70 threshold; no funding in preliminary economics, not fully net results; no holdout or tuning. Full-window universe selection bias."}
    try:audit(out)
    except Exception:out.update(status="fail",error=traceback.format_exc())
    p=ROOT/"reports"/"25_multi_asset_transfer.zip";p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(".tmp")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:z.writestr("metrics.json",json.dumps(out,allow_nan=False))
    tmp.replace(p);print(p,out["status"])
if __name__=="__main__":main()
