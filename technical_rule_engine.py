"""Fixed-horizon technical rule challenger screen, development only."""
import os
os.environ.setdefault("POLARS_MAX_THREADS","1")
import json,zipfile
from pathlib import Path
import numpy as np
import polars as pl
from robustness_engine import A,B,DAY,ledger,metrics
from execution_engine import score_trades
from baseline_engine import digest
from experiment_store import identity

ROOT=Path("data")
def main():
    folder,=list((ROOT/"experiments").glob("6823506d*"))
    spec=json.loads((folder/"spec.json").read_text())
    if identity(spec)!=folder.name:raise ValueError("Incumbent identity mismatch")
    manifest=json.loads((folder/"manifest.json").read_text())
    paths=sorted((folder/"predictions").glob("*.parquet"))
    if {p.stem for p in paths}!={str(f["fold_id"]) for f in manifest["folds"]}:raise ValueError("Incomplete predictions")
    f=pl.read_parquet(paths).sort("decision_time")
    if f["decision_time"].n_unique()!=f.height or f["symbol"].unique().to_list()!=["BTCUSDT"] or f["decision_time"].min()<A or f["label_end"].max()>=B:raise ValueError("Invalid scope")
    src=[p for p in sorted((ROOT/"resampled"/"BTCUSDT"/"1h").glob("*.parquet")) if "2020-12"<=p.stem<="2024-12"]
    x=pl.read_parquet(src).sort("open_time")
    if x["open_time"].n_unique()!=x.height:raise ValueError("Duplicate candles")
    if x.filter(pl.col("close_time")!=pl.col("open_time")+3600000-1).height:raise ValueError("Invalid candle boundary")
    x=x.with_columns((pl.col("open_time").diff().fill_null(3600000)!=3600000).cast(pl.Int64).cum_sum().alias("seg"),(pl.col("close_time")+1).alias("decision_time"))
    c=pl.col("close"); seg="seg"
    rules={}
    for fast,slow in [(9,21),(20,50),(50,200)]:
        a=c.rolling_mean(fast).over(seg); b=c.rolling_mean(slow).over(seg)
        rules[f"sma_{fast}_{slow}"]=pl.when(a>b).then(1).when(a<b).then(-1).otherwise(0)
    for n in (14,28):
        d=c.diff().over(seg)
        up=d.clip(lower_bound=0).rolling_mean(n).over(seg)
        dn=(-d).clip(lower_bound=0).rolling_mean(n).over(seg)
        # Simple rolling RSI, not Wilder smoothing.
        r=100*up/(up+dn)
        for lo in (30,40):
            rules[f"rsi_simple_{n}_{lo}"]=pl.when(r<lo).then(1).when(r>100-lo).then(-1).otherwise(0)
    for n in (20,50,100):
        hi=pl.col("high").shift(1).rolling_max(n).over(seg)
        low=pl.col("low").shift(1).rolling_min(n).over(seg)
        rules[f"channel_{n}"]=pl.when(c>hi).then(1).when(c<low).then(-1).otherwise(0)
        ret=c/c.shift(n).over(seg)-1
        for threshold in (.01,.02):
            rules[f"momentum_{n}_{threshold}"]=pl.when(ret>threshold).then(1).when(ret< -threshold).then(-1).otherwise(0)
    sig=x.select("decision_time",*[v.fill_null(0).cast(pl.Int8).alias(k) for k,v in rules.items()])
    f=f.join(sig,on="decision_time",how="left",validate="1:1")
    if any(f[k].null_count() for k in rules):raise ValueError("Missing hourly source for eligible decisions")
    p=f["prediction"].to_numpy()
    f=f.with_columns(pl.Series("incumbent",np.where(p>=.7,1,np.where(p<=.3,-1,0))),pl.lit(1).alias("always_long"),pl.lit(-1).alias("always_short"))
    fund=ledger(ROOT)
    missing=fund.filter(pl.col("mark_price").is_null())["event_time"].to_numpy()
    fund=fund.filter(pl.col("mark_price").is_not_null())
    first=f["decision_time"].min()//DAY*DAY;last=(f["label_end"].max()//DAY+1)*DAY
    results=[]
    for name in [*rules,"incumbent","always_long","always_short"]:
        sides=f[name].to_numpy(); ix=[]; free=-1
        for i,(d,e,s) in enumerate(zip(f["decision_time"],f["label_end"],sides)):
            if s and d>=free:ix.append(i);free=e
        t=f.select("symbol","decision_time","entry_time","label_end","entry_price","exit_price",pl.col(name).alias("side")).with_row_index("idx").filter(pl.col("idx").is_in(ix)).drop("idx")
        bad=np.searchsorted(missing,t["label_end"].to_numpy(),side="right")>np.searchsorted(missing,t["entry_time"].to_numpy(),side="right")
        total=t.height;t=t.filter(pl.Series("eligible",~bad))
        rows=[]
        for fee,slip in [(5,1),(10,5)]:
            scored=score_trades(t,fee,slip,fund,{"BTCUSDT":[(A,B)]}) if t.height else t
            rows.append({"fee_bps":fee,"slippage_bps":slip,**metrics(scored,first,last)})
        results.append({"rule":name,"original_trades":total,"excluded_unknown_funding":int(bad.sum()),"metrics":rows})
        print(name,"done",flush=True)
    out={"scope":"Exploratory fixed-horizon adaptations, not paper replication; common incumbent decision grid; non-overlap, 15m delay/4h exit; unknown funding excluded after selection; fixed notional not equity; no multiple testing correction","rules":list(rules),"source_hash":digest(src+paths+[Path(__file__),Path("execution_engine.py"),Path("robustness_engine.py")]),"results":results}
    p=ROOT/"reports"/"12_technical_rules.zip";p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(".tmp")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:z.writestr("metrics.json",json.dumps(out,allow_nan=False))
    tmp.replace(p);print(p)
if __name__=="__main__":main()
