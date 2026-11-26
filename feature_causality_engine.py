"""BTC feature replay and assumed auxiliary publication-lag sensitivity.
Development only; no writes to feature/label caches, no model fitting.
"""
import os
os.environ.setdefault("POLARS_MAX_THREADS","1")
from pathlib import Path
from datetime import timedelta
import json, zipfile, traceback
import numpy as np
import polars as pl
import data_engine as de
from feature_engine import technical, attach, read, partitions, ratio, BASE_FEATURES, METRICS, AUX_FEATURES, LOOKBACK

ROOT=Path("data")
S="BTCUSDT"
H=3600000
A=de.ms(de.date("2021-01-01"))
B=de.ms(de.date("2025-01-01"))

def same(a,b):
    return np.isclose(a,b,rtol=1e-8,atol=1e-10,equal_nan=True)

def clean(f,cols):
    return f.with_columns([pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c) for c in cols])

def compare(x,y,cols):
    if not x["decision_time"].equals(y["decision_time"]):
        raise ValueError("Decision grid mismatch")
    return {c:int((~same(x[c].cast(pl.Float64).to_numpy(),y[c].cast(pl.Float64).to_numpy())).sum()) for c in cols}

def audit(out, export=False):
    src=ROOT/"resampled"/S/"1h"
    paths=[p for p in sorted((ROOT/"features"/S/"1h").glob("*.parquet")) if "2021-01"<=p.stem<="2024-12"]
    if len(paths)!=48:raise ValueError("Expected 48 feature months")
    for p in paths:
        month=de.date(p.stem+"-01")
        a=de.ms(month); b=de.ms((month+timedelta(days=32)).replace(day=1))
        lo=a-(LOOKBACK+2)*H
        raw=read(partitions(src,lo,b),"open_time",lo,b)
        if raw is None:raise ValueError("Missing hourly source")
        if raw.filter((pl.col("close_time")!=pl.col("open_time")+H-1)|(pl.col("open_time")%H!=0)).height:raise ValueError("Invalid candle boundary")
        base=technical(raw,H).filter(pl.col("open_time").is_between(a,b,closed="left")).sort("decision_time")
        cached=pl.read_parquet(p).sort("decision_time")
        row={"month":p.stem,"rows":base.height,"technical_mismatches":compare(cached,clean(base,BASE_FEATURES),BASE_FEATURES),"prefix_mismatches":{},"lags":[]}
        out["months"].append(row)
        # Prefix invariance: recompute using only observations up to selected decisions.
        for i in sorted(set([0,base.height//2,base.height-1])):
            t=int(base["decision_time"][i])
            prefix=clean(technical(raw.filter(pl.col("close_time")<t),H),BASE_FEATURES).tail(1)
            row["prefix_mismatches"][str(t)]=compare(cached.filter(pl.col("decision_time")==t),prefix,BASE_FEATURES)
        aux={k:read(partitions(ROOT/k/S,a-7*86400000,b),"event_time",a-7*86400000,b) for k in ("metrics","funding","mark","index","spot","premium")}
        m=aux["metrics"]
        if m is not None and m.height:
            oi=pl.col("sum_open_interest")
            m=m.with_columns(pl.when(pl.col("event_time").diff()==300000).then(ratio(oi,oi.shift(1))-1).otherwise(None).alias("oi_change_5m"))
        original=None
        # Extra lag applied to all auxiliary sources, retaining original staleness tolerances.
        for extra in (0,300000,900000,3600000):
            x=attach(base,m,METRICS+["oi_change_5m"],300000+extra,300000,"metrics")
            x=x.with_columns(pl.when(pl.col("metrics_age_ms")==300000+extra).then(pl.col("oi_change_5m")).otherwise(None).alias("oi_change_5m"))
            x=attach(x,aux["funding"],["funding_rate"],60000+extra,24*H,"funding")
            for k in ("mark","index","spot","premium"):
                r=aux[k]; name=k+"_close"
                if r is not None:r=r.select("event_time",pl.col("close").alias(name))
                x=attach(x,r,[name],60000+extra,0,k)
            x=clean(x.with_columns(*[(ratio(pl.col("close"),pl.col(k+"_close")).log()).alias(k+"_basis") for k in ("mark","index","spot")],pl.col("premium_close").alias("premium")),AUX_FEATURES)
            if extra==0:
                original=x
                row["auxiliary_mismatches"]=compare(cached,x,AUX_FEATURES+[k+"_age_ms" for k in aux])
            if export and extra:
                target=ROOT/"lag_features"/str(extra//60000)/S/"1h"/p.name
                de.atomic_write(x.with_columns(pl.lit(S).alias("symbol")).select("symbol","decision_time",*BASE_FEATURES,*AUX_FEATURES),target)
            stats={}
            for c in AUX_FEATURES:
                v=x[c].cast(pl.Float64).to_numpy(); old=original[c].cast(pl.Float64).to_numpy()
                both=np.isfinite(v)&np.isfinite(old)
                stats[c]={"finite":int(np.isfinite(v).sum()),"both_finite":int(both.sum()),"changed_on_both":int((~same(v[both],old[both])).sum())}
            ages={}
            for k in aux:
                v=x[k+"_age_ms"].cast(pl.Float64).to_numpy(); v=v[np.isfinite(v)]
                lag=(300000 if k=="metrics" else 60000)+extra
                if np.any(v<lag):raise ValueError("Availability violation "+k)
                ages[k]={"min_ms":float(v.min()) if len(v) else None,"max_ms":float(v.max()) if len(v) else None}
            row["lags"].append({"extra_ms":extra,"features":stats,"ages":ages})
        print(p.stem,"audited",flush=True)
    out["status"]="fail" if any(sum(r["technical_mismatches"].values())+sum(r["auxiliary_mismatches"].values())+sum(sum(v.values()) for v in r["prefix_mismatches"].values()) for r in out["months"]) else "pass"

def main():
    out={"scope":"BTC development cached feature replay; sampled technical prefix invariance; assumed lag sensitivity only, not actual publication-time verification or economic sensitivity","months":[]}
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--export-lags",action="store_true")
    args=parser.parse_args()
    try:audit(out,args.export_lags)
    except Exception:
        out["status"]="fail";out["error"]=traceback.format_exc()
    path=ROOT/"reports"/"09_feature_causality.zip"
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:z.writestr("metrics.json",json.dumps(out,indent=2,allow_nan=False))
    tmp.replace(path);print(path,out["status"],flush=True)
    if out["status"]!="pass":raise SystemExit(out.get("error","Feature mismatches; inspect report"))
if __name__=="__main__":main()
