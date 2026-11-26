"""Frozen BTC logistic challenger: causal market regime features, development only."""
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
NEW=["vol_ratio_24_168","efficiency_24","efficiency_168","range_location_24","range_location_168"]
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
    # Regime features use completed hourly bars only, with no forward fill.
    raw=[ROOT/"resampled"/"BTCUSDT"/"1h"/p.name for p,q in pairs]
    source=digest([v for pair in pairs for v in pair]+raw)
    key=identity({"code":code,"source":source,"incumbent":folder.name,"features":NEW})
    cache=ROOT/"feature_v2"/key
    cache.mkdir(parents=True,exist_ok=True)
    y=pl.read_parquet(raw).sort("open_time")
    assert y["open_time"].n_unique()==y.height
    assert (y["close_time"]==y["open_time"]+H-1).all()
    y=y.with_columns((pl.col("open_time").diff().fill_null(H)!=H).cast(pl.Int64).cum_sum().alias("seg"),(pl.col("close_time")+1).alias("decision_time"))
    close=pl.col("close").log()
    y=y.with_columns(close.diff().over("seg").alias("r"))
    rr=pl.col("r")
    def ratio(n,d):return pl.when(d>0).then(n/d).otherwise(None)
    expr=[ratio(rr.rolling_std(24).over("seg"),rr.rolling_std(168).over("seg")).alias(NEW[0])]
    for n,k in [(24,1),(168,2)]:
        expr.append(ratio(rr.rolling_sum(n).over("seg"),rr.abs().rolling_sum(n).over("seg")).alias(NEW[k]))
    for n,k in [(24,3),(168,4)]:
        hi=pl.col("high").rolling_max(n).over("seg")
        lo=pl.col("low").rolling_min(n).over("seg")
        expr.append(ratio(pl.col("close")-lo,hi-lo).alias(NEW[k]))
    y=y.with_columns(*expr)
    x=x.join(y.select("decision_time",*NEW),on="decision_time",how="left",validate="1:1")
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
    for group,extra in [("market_regime",NEW)]:
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
    out={"scope":"Exploratory market regime adaptation; no tuning; incumbent calendar and eligibility; training-only median and scaling; completed hourly OHLC only; fixed-notional results, not portfolio"}
    try:main_audit(out)
    except Exception:out.update(status="fail",error=traceback.format_exc())
    p=ROOT/"reports"/"17_market_regime.zip";p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(".tmp")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:z.writestr("metrics.json",json.dumps(out,allow_nan=False))
    tmp.replace(p);print(p,out["status"])
if __name__=="__main__":main()
