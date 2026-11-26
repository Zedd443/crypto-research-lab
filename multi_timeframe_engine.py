"""Frozen BTC logistic challenger: causal completed higher-timeframe features, development only."""
import os
os.environ.setdefault("POLARS_MAX_THREADS","1")
import json,zipfile,traceback
from pathlib import Path
import numpy as np
import polars as pl
from baseline_engine import files,digest,load_fold,fit_score
from validation_protocol import Fold
from experiment_store import identity
from robustness_engine import A,B,DAY,ledger,pick,metrics
from funding_repair_engine import apply
from execution_engine import score_trades
from feature_engine import BASE_FEATURES
from data_engine import atomic_write,write_json
ROOT=Path("data")
H=3600000
NEW=["htf4h_ret_6","htf4h_sma_gap_20","htf4h_vol_20","htf1d_ret_7","htf1d_sma_gap_20","htf1d_vol_20"]
def main_audit(out):
    folder,=list((ROOT/"experiments").glob("6823506d*"))
    spec=json.loads((folder/"spec.json").read_text())
    man=json.loads((folder/"manifest.json").read_text())
    assert identity(spec)==folder.name and spec["manifest_id"]==man["manifest_id"]
    assert spec["model"]=="logistic" and spec["delay_minutes"]==15 and spec["horizon"]==4
    pairs=[(p,q) for p,q in files(ROOT,["BTCUSDT"],"1h",4,15) if "2021-01"<=p.stem<="2024-12"]
    assert len(pairs)==48
    cols=spec["features"]
    assert set(cols)==set(BASE_FEATURES)-{"body"}
    code=digest([Path(__file__),Path("baseline_engine.py"),Path("validation_protocol.py"),Path("feature_engine.py")])
    source=digest([v for pair in pairs for v in pair])
    key=identity({"code":code,"source":source,"incumbent":folder.name,"features":NEW})
    cache=ROOT/"feature_v2"/key
    cache.mkdir(parents=True,exist_ok=True)
    x=pl.read_parquet([p for p,q in pairs]).sort("decision_time")
    assert x["decision_time"].n_unique()==x.height
    # Build completed UTC-aligned higher-timeframe bars from verified hourly bars.
    raw=[ROOT/"resampled"/"BTCUSDT"/"1h"/p.name for p,q in pairs]
    source=digest([v for pair in pairs for v in pair]+raw)
    key=identity({"code":code,"source":source,"incumbent":folder.name,"features":NEW})
    cache=ROOT/"feature_v2"/key
    cache.mkdir(parents=True,exist_ok=True)
    y=pl.read_parquet(raw).sort("open_time")
    assert y["open_time"].n_unique()==y.height
    assert (y["close_time"]==y["open_time"]+H-1).all()
    assert (y["open_time"]%H==0).all()
    for hours,n,tag in [(4,6,"htf4h"),(24,7,"htf1d")]:
        step=hours*H
        bars=y.with_columns((pl.col("open_time")//step*step).alias("start")).group_by("start").agg(pl.len().alias("count"),pl.col("close").sort_by("open_time").last().alias("close")).filter(pl.col("count")==hours).sort("start")
        bars=bars.with_columns((pl.col("start")+step).alias("available"),(pl.col("start").diff().fill_null(step)!=step).cast(pl.Int64).cum_sum().alias("seg"))
        c=pl.col("close")
        bars=bars.with_columns(c.log().diff().over("seg").alias("r"))
        names=[f"{tag}_ret_{n}",f"{tag}_sma_gap_20",f"{tag}_vol_20"]
        bars=bars.with_columns((c.log()-c.shift(n).over("seg").log()).alias(names[0]),(c/c.rolling_mean(20).over("seg")-1).alias(names[1]),pl.col("r").rolling_std(20).over("seg").alias(names[2]))
        # No as-of carry across a missing higher-timeframe period.
        x=x.sort("decision_time").join_asof(bars.select("available",*names),left_on="decision_time",right_on="available",strategy="backward",tolerance=step-1)
        assert x.filter(pl.col("available")>pl.col("decision_time")).height==0
        x=x.drop("available")
    x=x.with_columns(*[pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c) for c in NEW])
    # Preserve the exact original monthly partition membership via symbol/time join.
    derived=[]
    for p,q in pairs:
        dst=cache/"features"/p.name
        if not dst.exists():
            base=pl.read_parquet(p)
            atomic_write(base.join(x.select("symbol","decision_time",*NEW),on=["symbol","decision_time"],how="left",validate="1:1"),dst)
        derived.append((dst,q))
    out.update({"experiment":key,"source":source,"code":code,"features":NEW,"missing":{c:x[c].null_count() for c in NEW},"rows":x.height,"results":[]})
    fund=apply(ledger(ROOT)).sort("event_time")
    missing=fund.filter(pl.col("mark_price").is_null())["event_time"].to_numpy()
    fund=fund.filter(pl.col("mark_price").is_not_null())
    params=json.loads((folder/"result.json").read_text())["params"]
    for group,extra in [("multi_timeframe",NEW)]:
        paths=[]
        for fd in man["folds"]:
            fold=Fold(**fd)
            p=cache/group/(str(fold.fold_id)+".parquet")
            if not p.exists():
                train,test=load_fold(derived,cols+extra,fold)
                fit_score(train,test,cols+extra,"logistic",params,1,spec["seed"],p)
            old=pl.read_parquet(folder/"predictions"/p.name)
            new=pl.read_parquet(p)
            keys=["symbol","decision_time","entry_time","label_end","entry_price","exit_price","target_return"]
            assert new.select(keys).equals(old.select(keys)),"OOS eligibility mismatch"
            paths.append(p);print(group,fold.fold_id,"done",flush=True)
        for name,ps in [("incumbent",[folder/"predictions"/p.name for p in paths]),(group,paths)]:
            f=pl.read_parquet(ps).sort("decision_time")
            assert f["label_end"].max()<B and f["decision_time"].min()>=A
            y=f["target_return"].to_numpy()>0
            prob=f["prediction"].to_numpy()
            t=pick(f,"logistic",.7)
            bad=np.searchsorted(missing,t["label_end"].to_numpy(),side="right")>np.searchsorted(missing,t["entry_time"].to_numpy(),side="right")
            assert not bad.any(),"Unrecovered funding in selected trades"
            rows=[]
            first=f["decision_time"].min()//DAY*DAY;last=(f["label_end"].max()//DAY+1)*DAY
            for fee,slip in [(5,1),(10,5)]:
                s=score_trades(t,fee,slip,fund,{"BTCUSDT":[(A,B)]}) if t.height else t
                rows.append({"fee_bps":fee,"slippage_bps":slip,**metrics(s,first,last)})
            out["results"].append({"name":name,"brier":float(np.mean((prob-y)**2)),"naive_brier":float(np.mean((f["baseline_prediction"].to_numpy()-y)**2)),"DA":float(np.mean((prob>=.5)==y)),"economics":rows})
    out["status"]="pass"
def main():
    out={"scope":"Exploratory completed higher-timeframe adaptation; no tuning; incumbent calendar and eligibility; training-only median and scaling; completed UTC-aligned 4h and daily bars from hourly source; causal as-of with bounded age; fixed-notional results, not portfolio"}
    try:main_audit(out)
    except Exception:out.update(status="fail",error=traceback.format_exc())
    p=ROOT/"reports"/"20_multi_timeframe.zip";p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(".tmp")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:z.writestr("metrics.json",json.dumps(out,allow_nan=False))
    tmp.replace(p);print(p,out["status"])
if __name__=="__main__":main()
