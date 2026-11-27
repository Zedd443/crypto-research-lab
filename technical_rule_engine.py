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
    import argparse
    from funding_repair_engine import apply
    parser=argparse.ArgumentParser();parser.add_argument("--extended",action="store_true");parser.add_argument("--stateful",action="store_true");args=parser.parse_args()
    if args.stateful:args.extended=True
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
    if args.extended:
        rules={}
        # Rolling signed volume is OBV minus its n-bar lag, reset at gaps.
        signed=pl.when(c.diff().over(seg)>0).then(pl.col("volume")).when(c.diff().over(seg)<0).then(-pl.col("volume")).otherwise(0.)
        x=x.with_columns(signed.alias("signed_obv"))
        for n in (20,100):
            flow=pl.col("signed_obv").rolling_sum(n).over(seg)
            rules[f"obv_{n}"]=pl.when(flow>0).then(1).when(flow<0).then(-1).otherwise(0)
        # Fixed percentage filter: reverse from running closing-price extrema.
        for threshold in (.01,.02):
            values=[];pos=0;hi=lo=None;prev=None
            for price,segment in x.select("close","seg").iter_rows():
                if segment!=prev:pos=0;hi=lo=price;prev=segment
                hi=max(hi,price);lo=min(lo,price)
                if pos<=0 and price>=lo*(1+threshold):pos=1;hi=price
                elif pos>=0 and price<=hi*(1-threshold):pos=-1;lo=price
                values.append(pos)
            name=f"filter_{threshold}"
            x=x.with_columns(pl.Series(name,values));rules[name]=pl.col(name)
    sig=x.select("decision_time",*[v.fill_null(0).cast(pl.Int8).alias(k) for k,v in rules.items()])
    f=f.join(sig,on="decision_time",how="left",validate="1:1")
    if any(f[k].null_count() for k in rules):raise ValueError("Missing hourly source for eligible decisions")
    p=f["prediction"].to_numpy()
    f=f.with_columns(pl.Series("incumbent",np.where(p>=.7,1,np.where(p<=.3,-1,0))),pl.lit(1).alias("always_long"),pl.lit(-1).alias("always_short"))
    fund=apply(ledger(ROOT)) if args.extended else ledger(ROOT)
    missing=fund.filter(pl.col("mark_price").is_null())["event_time"].to_numpy()
    fund=fund.filter(pl.col("mark_price").is_not_null())
    first=f["decision_time"].min()//DAY*DAY;last=(f["label_end"].max()//DAY+1)*DAY
    results=[]
    for name in (list(rules) if args.stateful else ([*rules,"incumbent"] if args.extended else [*rules,"incumbent","always_long","always_short"])):
        sides=f[name].to_numpy(); ix=[]; free=-1
        for i,(d,e,s) in enumerate(zip(f["decision_time"],f["label_end"],sides)):
            if s and d>=free:ix.append(i);free=e
        t=f.select("symbol","decision_time","entry_time","label_end","entry_price","exit_price",pl.col(name).alias("side")).with_row_index("idx").filter(pl.col("idx").is_in(ix)).drop("idx")
        if args.stateful:
            dt=f["decision_time"].to_numpy();et=f["entry_time"].to_numpy();ep=f["entry_price"].to_numpy()
            if not np.all(et-dt==900000):raise ValueError("Unexpected execution delay")
            records=[];pos=0;start=None
            for i,v in enumerate(sides):
                desired=int(v) if i<len(sides)-1 else 0
                if desired==pos:continue
                if pos:records.append({"symbol":"BTCUSDT","decision_time":int(dt[start]),"entry_time":int(et[start]),"label_end":int(et[i]),"entry_price":float(ep[start]),"exit_price":float(ep[i]),"side":pos})
                pos=desired;start=i if pos else None
            t=pl.DataFrame(records,schema={"symbol":pl.String,"decision_time":pl.Int64,"entry_time":pl.Int64,"label_end":pl.Int64,"entry_price":pl.Float64,"exit_price":pl.Float64,"side":pl.Int64})
        bad=np.searchsorted(missing,t["label_end"].to_numpy(),side="right")>np.searchsorted(missing,t["entry_time"].to_numpy(),side="right")
        if args.stateful and np.any(bad):raise ValueError("Unknown funding affects stateful trades")
        total=t.height;t=t.filter(pl.Series("eligible",~bad))
        rows=[]
        for fee,slip in [(5,1),(10,5)]:
            scored=score_trades(t,fee,slip,fund,{"BTCUSDT":[(A,B)]}) if t.height else t
            rows.append({"fee_bps":fee,"slippage_bps":slip,**metrics(scored,first,last)})
        duration=(t["label_end"]-t["entry_time"]).to_numpy()/3600000
        results.append({"holding_hours_mean":float(duration.mean()) if len(duration) else None,"exposure":float(duration.sum()*3600000/(last-first)),"rule":name,"original_trades":total,"excluded_unknown_funding":int(bad.sum()),"metrics":rows})
        print(name,"done",flush=True)
    out={"scope":"Exploratory fixed-horizon adaptations, not paper replication; common incumbent decision grid; non-overlap, 15m delay/4h exit; unknown funding excluded after selection; fixed notional not equity; no multiple testing correction","rules":list(rules),"source_hash":digest(src+paths+[Path(__file__),Path("execution_engine.py"),Path("robustness_engine.py")]),"results":results}
    if args.extended:out["scope"]+="; OBV signed rolling volume and percentage filter state adaptations; recovered funding ledger; fixed 4h exits, not stateful execution";out["status"]="pass"
    if args.stateful:
        out["scope"]="Stateful OBV/filter adaptations on incumbent eligible grid, 15m execution delay; hold until signal changes, flatten on neutral and close at terminal eligible execution time; gap may delay exit; complete recovered funding, fixed notional; no tuning or holdout; not exact paper replication"
        out["decision_gap_count"]=int(np.sum(np.diff(f["decision_time"].to_numpy())>3600000))
        out["max_decision_gap_hours"]=float(np.max(np.diff(f["decision_time"].to_numpy()))/3600000)
    p=ROOT/"reports"/("43_stateful_extended.zip" if args.stateful else ("38_technical_extended.zip" if args.extended else "12_technical_rules.zip"));p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(".tmp")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:z.writestr("metrics.json",json.dumps(out,allow_nan=False))
    tmp.replace(p);print(p)
if __name__=="__main__":main()
