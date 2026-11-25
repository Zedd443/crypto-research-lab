"""Point-in-time technical/derivatives features with partition-level cache.
Labels are stored separately. All times are UTC milliseconds.
"""
import argparse
import os
import multiprocessing as mp
os.environ.setdefault('POLARS_MAX_THREADS','1')
from datetime import timedelta
import json
import hashlib
from pathlib import Path
import polars as pl
import data_engine as de
from universe import SYMBOLS

VERSION=hashlib.sha256(Path(__file__).read_bytes()+Path(de.__file__).read_bytes()+pl.__version__.encode()).hexdigest()
STEPS={"15m":900000,"1h":3600000,"4h":14400000,"1d":86400000}
LOOKBACK=200
HORIZONS_H=(1,4,8,24)
METRICS=["sum_open_interest","sum_open_interest_value","count_toptrader_long_short_ratio",
         "sum_toptrader_long_short_ratio","count_long_short_ratio","sum_taker_long_short_vol_ratio"]
BASE_FEATURES=["ret_1","ret_4","ret_16","sma_gap_20","sma_gap_200","volatility_20",
               "relative_volume_20","volume_delta","flow_20","body","range","trade_size"]
AUX_FEATURES=METRICS+["oi_change_5m","funding_rate","mark_basis","index_basis","spot_basis","premium"]
FEATURES=BASE_FEATURES+AUX_FEATURES

def ratio(a,b):
    return pl.when(b.is_not_null() & (b!=0)).then(a/b).otherwise(None)

def technical(f,step):
    f=f.sort("open_time")
    if f["open_time"].n_unique()!=f.height:
        raise ValueError("Duplicate feature input timestamps")
    f=f.with_columns((pl.col("open_time").diff().fill_null(step)!=step).cast(pl.Int64).cum_sum().alias("segment"))
    c=pl.col("close"); v=pl.col("volume")
    f=f.with_columns((2*pl.col("taker_buy_base_volume")-v).alias("signed_volume"),
        *[(c.log()-c.shift(n).over("segment").log()).alias("ret_"+str(n)) for n in (1,4,16)])
    return f.with_columns(
        *[(ratio(c,c.rolling_mean(n).over("segment"))-1).alias("sma_gap_"+str(n)) for n in (20,200)],
        pl.col("ret_1").rolling_std(20).over("segment").alias("volatility_20"),
        ratio(v,v.rolling_mean(20).over("segment")).alias("relative_volume_20"),
        ratio(pl.col("signed_volume"),v).alias("volume_delta"),
        ratio(pl.col("signed_volume").rolling_sum(20).over("segment"),v.rolling_sum(20).over("segment")).alias("flow_20"),
        ratio(c-pl.col("open"),pl.col("open")).alias("body"),
        ratio(pl.col("high")-pl.col("low"),c).alias("range"),
        ratio(pl.col("quote_volume"),pl.col("trades")).alias("trade_size"),
        (pl.col("close_time")+1).alias("decision_time"))

def labels(f,step,hours):
    """One full bar latency after signal; open-to-open targets; gaps invalidate."""
    h=hours*3600000
    if h%step:raise ValueError("Horizon must be a multiple of timeframe")
    n=h//step
    f=f.sort("open_time")
    if f["open_time"].n_unique()!=f.height or f.filter(pl.col("open_time")%step!=0).height:
        raise ValueError("Invalid label timestamp grid")
    t=pl.col("open_time")
    # Signal at bar close; enter one bar later, exit n bars after entry.
    return f.select((pl.col("close_time")+1).alias("decision_time"),
        t.shift(-2).alias("entry_time"),t.shift(-(n+2)).alias("label_end"),
        pl.col("open").shift(-2).alias("entry_price"),
        pl.col("open").shift(-(n+2)).alias("exit_price")).filter(
        (pl.col("entry_time")==pl.col("decision_time")+step)&
        (pl.col("label_end")==pl.col("decision_time")+(n+1)*step)&
        (pl.col("entry_price")>0)&(pl.col("exit_price")>0)).with_columns(
        (pl.col("exit_price")/pl.col("entry_price")).log().alias("target_return"))

def partitions(directory,a,b):
    result=[]
    for p in sorted(directory.glob("*.parquet")):
        month=de.date(p.stem+"-01")
        nxt=(month+timedelta(days=32)).replace(day=1)
        if de.ms(month)<b and de.ms(nxt)>a:result.append(p)
    return result

def read(paths,key,a,b):
    if not paths:return None
    return pl.scan_parquet(paths).filter(pl.col(key).is_between(a,b,closed="left")).collect().sort(key)

def attach(left,right,cols,lag,tolerance,tag):
    if right is None or not right.height:
        return left.with_columns(*[pl.lit(None,dtype=pl.Float64).alias(c) for c in cols],
                                 pl.lit(None,dtype=pl.Int64).alias(tag+"_age_ms"))
    if right["event_time"].n_unique()!=right.height:
        raise ValueError("Duplicate auxiliary observations: "+tag)
    key=tag+"_available_time"
    right=right.select((pl.col("event_time")+lag).alias(key),*[pl.col(c).cast(pl.Float64) for c in cols]).sort(key)
    out=left.sort("decision_time").join_asof(right,left_on="decision_time",right_on=key,
        strategy="backward",tolerance=tolerance)
    return out.with_columns((pl.col("decision_time")-pl.col(key)+lag).alias(tag+"_age_ms")).drop(key)

def build(root,symbol,tf,force=False):
    step=STEPS[tf]
    src=root/"resampled"/symbol/tf
    horizons=[h for h in HORIZONS_H if h*3600000%step==0]
    for p in sorted(src.glob("*.parquet")):
        month=de.date(p.stem+"-01")
        a=de.ms(month); b=de.ms((month+timedelta(days=32)).replace(day=1))
        lo=a-(LOOKBACK+2)*step
        hi=b+max(horizons)*3600000+2*step
        prices=partitions(src,lo,hi)
        aux={k:partitions(root/k/symbol,a-7*86400000,b) for k in ("metrics","funding","mark","index","spot","premium")}
        dependencies={str(x.relative_to(root)):de.signature(x) for x in [*prices,*[p for v in aux.values() for p in v]]}
        base=root/"features"/symbol/tf
        targets=[base/p.name]+[root/"labels"/symbol/tf/str(h)/p.name for h in horizons]
        marker=base/(p.stem+".json")
        try:old=json.loads(marker.read_text())
        except (OSError,ValueError):old={}
        if not force and old.get("version")==VERSION and old.get("dependencies")==dependencies and all(
            x.exists() and old.get("outputs",{}).get(str(x.relative_to(root)))==de.signature(x) for x in targets):
            print(symbol,tf,p.stem,"feature cached",flush=True);continue
        f=read(prices,"open_time",lo,hi)
        if f is None or not f.height:continue
        if f["open_time"].n_unique()!=f.height:raise ValueError("Duplicate source candles")
        left=technical(f.filter(pl.col("open_time")<b),step).filter(pl.col("open_time").is_between(a,b,closed="left"))
        left=left.with_columns(pl.lit(symbol).alias("symbol"))
        m=read(aux["metrics"],"event_time",a-7*86400000,b)
        if m is not None and m.height:
            oi=pl.col("sum_open_interest")
            m=m.with_columns(pl.when(pl.col("event_time").diff()==300000).then(ratio(oi,oi.shift(1))-1).otherwise(None).alias("oi_change_5m"))
        # Conservative 5m publication lag; at most one extra 5m stale interval.
        left=attach(left,m,METRICS+["oi_change_5m"],300000,300000,"metrics")
        left=left.with_columns(pl.when(pl.col("metrics_age_ms")==300000).then(pl.col("oi_change_5m")).otherwise(None).alias("oi_change_5m"))
        funding=read(aux["funding"],"event_time",a-7*86400000,b)
        left=attach(left,funding,["funding_rate"],60000,24*3600000,"funding")
        for k in ("mark","index","spot","premium"):
            r=read(aux[k],"event_time",a-7*86400000,b)
            name=k+"_close"
            if r is not None:r=r.select("event_time",pl.col("close").alias(name))
            left=attach(left,r,[name],60000,0,k)
        left=left.with_columns(*[(ratio(pl.col("close"),pl.col(k+"_close")).log()).alias(k+"_basis") for k in ("mark","index","spot")],
                               pl.col("premium_close").alias("premium"))
        left=left.with_columns(*[pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c) for c in FEATURES])
        # TradFi scheduled closures and real missing periods both break continuity.
        de.atomic_write(left.select("symbol","decision_time",*FEATURES,*[c for c in left.columns if c.endswith("_age_ms")]),targets[0])
        for h,target in zip(horizons,targets[1:]):
            l=labels(f,step,h).filter(pl.col("decision_time").is_between(a+step,b+step,closed="left")).with_columns(pl.lit(symbol).alias("symbol"))
            de.atomic_write(l,target)
        de.write_json(marker,{"version":VERSION,"dependencies":dependencies,"outputs":{str(x.relative_to(root)):de.signature(x) for x in targets},
                              "assumptions":"Auxiliary publication lags assumed conservatively, not verified historical publication timestamps. No future fill."})
        print(symbol,tf,p.stem,left.height,"features",flush=True)

def build_task(task):
    build(*task)
    return task[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path("data"))
    p.add_argument("--symbols",nargs="+",choices=SYMBOLS,default=SYMBOLS)
    p.add_argument("--tf",choices=list(STEPS),default="1h")
    p.add_argument("--force",action="store_true")
    p.add_argument("--workers",type=int,choices=range(1,9),default=2)
    a=p.parse_args()
    from resource_policy import budget
    tasks=[(a.root,s,a.tf,a.force) for s in dict.fromkeys(a.symbols)]
    workers=min(len(tasks),budget(a.workers,per_worker_bytes=1024**3))
    print("Feature workers:",workers,flush=True)
    if workers==1:
        for task in tasks:build_task(task)
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            for symbol in pool.imap_unordered(build_task,tasks,chunksize=1):print(symbol,"DONE",flush=True)
if __name__=="__main__":
    mp.freeze_support()
    main()
