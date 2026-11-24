"""E00b: archive-first auxiliary Binance datasets, bounded by local 1m coverage."""
import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
import polars as pl
from data_engine import SESSION, BASE, API, UTC, ms, date, months, days, atomic_write, MINUTE, gaps, clean, aggregate
from universe import SYMBOLS

KINDS={"mark":"markPriceKlines","index":"indexPriceKlines","premium":"premiumIndexKlines","metrics":"metrics","funding":"fundingRate","spot":"klines"}
METRICS=["sum_open_interest","sum_open_interest_value","count_toptrader_long_short_ratio","sum_toptrader_long_short_ratio","count_long_short_ratio","sum_taker_long_short_vol_ratio"]
def read_zip(url):
    r=SESSION.get(url,timeout=90)
    if r.status_code==404: return None
    r.raise_for_status()
    c=SESSION.get(url+".CHECKSUM",timeout=30); c.raise_for_status()
    if hashlib.sha256(r.content).hexdigest()!=c.text.split()[0].lower(): raise ValueError("Checksum mismatch "+url)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        names=[n for n in z.namelist() if n.endswith(".csv")]
        if len(names)!=1: raise ValueError("Unexpected archive members")
        return list(csv.reader(io.StringIO(z.read(names[0]).decode("utf-8-sig"))))

def parse(kind,rows):
    if not rows: return None
    if kind=="metrics":
        header=[v.strip().lower() for v in rows[0]]
        if "create_time" not in header: raise ValueError("Unknown metrics schema")
        f=pl.DataFrame(rows[1:],schema=header,orient="row")
        return f.select(pl.col("create_time").str.to_datetime(time_zone="UTC").dt.epoch("ms").alias("event_time"),
                        *[pl.col(c).cast(pl.Float64,strict=False) for c in METRICS])
    if kind=="funding":
        header=[v.strip().lower() for v in rows[0]]
        if "calc_time" in header:
            f=pl.DataFrame(rows[1:],schema=header,orient="row")
            return f.select(pl.col("calc_time").cast(pl.Int64).alias("event_time"),
                            pl.col("last_funding_rate").cast(pl.Float64).alias("funding_rate"),
                            pl.col("funding_interval_hours").cast(pl.Int64))
        raise ValueError("Unknown funding schema")
    rows=[r for r in rows if r and r[0].isdigit()]
    if not rows:return None
    f=pl.DataFrame([r[:5] for r in rows],schema=["event_time","open","high","low","close"],orient="row").cast({"event_time":pl.Int64,"open":pl.Float64,"high":pl.Float64,"low":pl.Float64,"close":pl.Float64})
    # Spot archive switched to microsecond timestamps in 2025.
    return f.with_columns(pl.when(pl.col("event_time")>10**14).then(pl.col("event_time")//1000).otherwise(pl.col("event_time")).alias("event_time"))

def fetch(kind,symbol,period,stamp):
    family=KINDS[kind]
    base="https://data.binance.vision/data/spot" if kind=="spot" else BASE
    interval="" if kind in ("metrics","funding") else "/1m"
    tag="" if kind in ("metrics","funding") else "-1m"
    name=family if kind in ("metrics","funding") else ""
    filename=f"{symbol}-{name+'-' if name else ''}{tag.lstrip('-')+'-' if tag else ''}{stamp}.zip"
    url=f"{base}/{period}/{family}/{symbol}{interval}/{filename}"
    return parse(kind,read_zip(url))

def coverage(root,symbol):
    paths=sorted((root/"futures"/symbol/"1m").glob("*.parquet"))
    for p in paths:
        f=pl.read_parquet(p,columns=["open_time"])
        if f.height: yield p, f["open_time"].min(),f["open_time"].max()+MINUTE

def download(root,symbol,kind,start,end):
    for p,first,last in coverage(root,symbol):
        a,b=max(first,ms(start)),min(last,ms(end))
        if a>=b:continue
        target=root/kind/symbol/p.name
        marker=target.with_suffix(".json")
        if target.exists() and marker.exists():
            meta=json.loads(marker.read_text())
            if meta.get("requested")==[a,b] and meta.get("complete"): 
                print(symbol,kind,p.stem,"cached");continue
        parts=[]
        missing=[]
        if target.exists(): parts.append(pl.read_parquet(target))
        if kind!="metrics":
            f=fetch(kind,symbol,"monthly",p.stem)
            if f is not None: parts.append(f)
        else: f=None
        if f is None and kind!="funding":
            for d in days(datetime.fromtimestamp(a/1000,UTC),datetime.fromtimestamp(b/1000,UTC)):
                daily=fetch(kind,symbol,"daily",d.strftime("%Y-%m-%d"))
                if daily is None:missing.append(d.strftime("%Y-%m-%d"))
                else:parts.append(daily)
        if kind=="funding" and f is None:
            cursor=a
            while cursor<b:
                r=SESSION.get(API+"/fapi/v1/fundingRate",params={"symbol":symbol,"startTime":cursor,"endTime":b-1,"limit":1000},timeout=30)
                if r.status_code==400:break
                r.raise_for_status()
                page=r.json()
                if not page:break
                parts.append(pl.DataFrame([{"event_time":int(x["fundingTime"]),"funding_rate":float(x["fundingRate"]),"funding_interval_hours":None} for x in page],schema={"event_time":pl.Int64,"funding_rate":pl.Float64,"funding_interval_hours":pl.Int64}))
                nxt=int(page[-1]["fundingTime"])+1
                if nxt<=cursor:raise ValueError("Funding cursor stalled")
                cursor=nxt
        count=0
        complete=False
        if parts:
            merged=pl.concat(parts,how="vertical_relaxed").unique().sort("event_time")
            if merged["event_time"].n_unique()!=merged.height:raise ValueError("Conflicting timestamps")
            atomic_write(merged,target)
            requested=merged.filter(pl.col("event_time").is_between(a,b,closed="left"))
            count=requested.height
            if kind in ("mark","index","premium","spot"):
                complete=(count==(b-a)//MINUTE and requested["event_time"].min()==a and requested["event_time"].max()==b-MINUTE and requested["event_time"].diff().drop_nulls().eq(MINUTE).all())
            # Event and metrics coverage cannot be inferred safely from count alone.
        marker.parent.mkdir(parents=True,exist_ok=True)
        marker.write_text(json.dumps({"requested":[a,b],"rows":count,"missing_archive_days":missing,"complete":complete,"note":"Coverage not guaranteed; no imputation. Funding and metrics require separate schedule audit."},indent=2))
        print(symbol,kind,p.stem,count,"rows",flush=True)

def audit(root,symbol):
    records=[]
    for p,a,b in coverage(root,symbol):
        f=clean(pl.read_parquet(p),a,b)
        missing=list(gaps(f,a,b))
        records.append({"month":p.stem,"first":a,"last_exclusive":b,"rows":f.height,"internal_gaps":missing})
    out=root/"metadata"/symbol/"coverage.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(records,indent=2))
    print(symbol,len(records),"partitions",sum(r["rows"] for r in records),"candles")
def main():
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=["download","audit"])
    p.add_argument("--symbols",nargs="+",choices=SYMBOLS,default=SYMBOLS)
    p.add_argument("--kinds",nargs="+",choices=list(KINDS),default=["mark","index","premium","metrics","funding"])
    p.add_argument("--root",type=Path,default=Path("data"))
    p.add_argument("--start",default="2019-09-01")
    p.add_argument("--end",default=datetime.now(UTC).strftime("%Y-%m-%d"))
    a=p.parse_args()
    failed=[]
    for s in a.symbols:
        if a.command=="audit":audit(a.root,s);continue
        for k in a.kinds:
            try:download(a.root,s,k,date(a.start),date(a.end))
            except Exception as e:
                print(s,k,"FAILED",repr(e),flush=True);failed.append((s,k))
    if failed:raise SystemExit(str(failed))
if __name__=="__main__":main()
