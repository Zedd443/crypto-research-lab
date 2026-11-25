"""Partitioned, UTC Binance USD-M kline ingestion. No synthetic gap filling."""
import argparse
import calendar
import hashlib
import io
import json
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from universe import SYMBOLS, TIMEFRAMES

BASE = "https://data.binance.vision/data/futures/um"
API = "https://fapi.binance.com"
FIELDS = ["open_time","open","high","low","close","volume","close_time",
          "quote_volume","trades","taker_buy_base_volume","taker_buy_quote_volume"]
MINUTE = 60_000
UTC = timezone.utc
SESSION = requests.Session()
SESSION.mount("https://", HTTPAdapter(max_retries=Retry(total=4, backoff_factor=1,
    status_forcelist=[429,500,502,503,504], respect_retry_after_header=True)))

def ms(d):
    return int(d.timestamp()*1000)

def date(s):
    return datetime.fromisoformat(s).replace(tzinfo=UTC)

def months(start,end):
    cur = start.replace(day=1,hour=0,minute=0,second=0,microsecond=0)
    while cur < end:
        nxt = (cur + timedelta(days=32)).replace(day=1)
        yield cur,nxt
        cur = nxt

def days(start,end):
    cur = start.replace(hour=0,minute=0,second=0,microsecond=0)
    while cur < end:
        yield cur
        cur += timedelta(days=1)

def parse_rows(rows):
    if not rows:
        return pl.DataFrame(schema={k:pl.Int64 if k in ("open_time","close_time","trades") else pl.Float64 for k in FIELDS})
    return pl.DataFrame([r[:11] for r in rows], schema=FIELDS, orient="row").cast(
        {k:pl.Int64 if k in ("open_time","close_time","trades") else pl.Float64 for k in FIELDS})

def archive(symbol,period,stamp):
    url = f"{BASE}/{period}/klines/{symbol}/1m/{symbol}-1m-{stamp}.zip"
    r = SESSION.get(url,timeout=90)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    check = SESSION.get(url+".CHECKSUM",timeout=30)
    check.raise_for_status()
    if hashlib.sha256(r.content).hexdigest() != check.text.split()[0].lower():
        raise ValueError("Checksum mismatch: "+url)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if len(names)!=1:
            raise ValueError("Unexpected archive members")
        b = z.read(names[0])
    import csv
    return parse_rows([r for r in csv.reader(io.StringIO(b.decode("utf-8-sig"))) if r and r[0].isdigit()])

def rest(symbol,start,end):
    rows=[]
    while start<end:
        r=SESSION.get(API+"/fapi/v1/klines",params={"symbol":symbol,"interval":"1m",
            "startTime":start,"endTime":end-1,"limit":1000},timeout=30)
        if r.status_code==400:
            body=r.json()
            if body.get("code")==-1121:
                return parse_rows(rows)
        r.raise_for_status()
        page=r.json()
        if not page:
            break
        rows.extend(page)
        nxt=int(page[-1][0])+MINUTE
        if nxt<=start:
            raise ValueError("REST cursor stalled")
        start=nxt
        time.sleep(.1)
    return parse_rows(rows)

def clean(frame,start,end):
    f=frame.filter((pl.col("open_time")>=start)&(pl.col("open_time")<end))
    f=f.unique().sort("open_time")
    if f["open_time"].n_unique()!=f.height:
        raise ValueError("Conflicting duplicate timestamps")
    bad=f.filter(pl.any_horizontal([pl.col(c).is_null() | ~pl.col(c).is_finite() for c in FIELDS]) |
        (pl.col("open_time")%MINUTE!=0) | (pl.col("close_time")!=pl.col("open_time")+MINUTE-1) |
        (pl.min_horizontal("open","high","low","close")<=0) |
        (pl.col("high")<pl.max_horizontal("open","low","close")) |
        (pl.col("low")>pl.min_horizontal("open","high","close")) |
        (pl.min_horizontal("volume","quote_volume","trades","taker_buy_base_volume","taker_buy_quote_volume")<0) |
        (pl.col("taker_buy_base_volume")>pl.col("volume")+1e-8) |
        (pl.col("taker_buy_quote_volume")>pl.col("quote_volume")+1e-8))
    if bad.height:
        raise ValueError(f"{bad.height} invalid candles")
    return f

def gaps(f,start,end):
    cursor=start
    for t in f["open_time"]:
        if t>cursor:
            yield cursor,t
        cursor=t+MINUTE
    if cursor<end:
        yield cursor,end

def atomic_write(f,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    f.write_parquet(tmp,compression="zstd")
    tmp.replace(path)

def cache_valid(report,path,old,a,b):
    # Only closed historical partitions; current month always refreshes.
    if not report.exists() or not path.exists() or not old.height:
        return False
    try:
        m=json.loads(report.read_text())
        x,y=m["requested"]
        if not (x<=a and b<=y) or m["rows"]!=old.height:
            return False
        if m.get("retrieval_finished") is False:
            return False
        # Legacy reports were written only after archive retrieval finished.
        actual=list(gaps(clean(old,x,y),x,y))
        return actual==[tuple(v) for v in m["missing_ranges_ms"]]
    except (ValueError,KeyError,TypeError,OSError):
        return False

def download(root,symbol,start,end,repair=False):
    for month,nxt in months(start,end):
        a,b=ms(max(start,month)),ms(min(end,nxt))
        path=root/"futures"/symbol/"1m"/(month.strftime("%Y-%m")+".parquet")
        old=pl.read_parquet(path) if path.exists() else parse_rows([])
        f=clean(old,a,b)
        missing=list(gaps(f,a,b))
        report=root/"metadata"/symbol/(month.strftime("%Y-%m")+".json")
        historical=nxt<=datetime.now(UTC).replace(day=1,hour=0,minute=0,second=0,microsecond=0)
        if not missing:
            print(symbol,month.date(),"cached",flush=True)
            continue
        if historical and not repair and cache_valid(report,path,old,a,b):
            print(symbol,month.date(),"cached (retrieval; see quality audit)",flush=True)
            continue
        parts=[old]
        full=archive(symbol,"monthly",month.strftime("%Y-%m")) if nxt<=datetime.now(UTC) else None
        if full is not None:
            parts.append(full)
        else:
            wanted=sorted({d for x,y in missing for d in days(datetime.fromtimestamp(x/1000,UTC),datetime.fromtimestamp(y/1000,UTC))})
            for d in wanted:
                daily=archive(symbol,"daily",d.strftime("%Y-%m-%d"))
                if daily is not None:
                    parts.append(daily)
                elif d>=datetime.now(UTC)-timedelta(days=7):
                    parts.append(rest(symbol,max(a,ms(d)),min(b,ms(d+timedelta(days=1)))))
        # Keep rows outside a user-requested subrange when updating a partition.
        merged=clean(pl.concat(parts),ms(month),min(ms(nxt),ms(datetime.now(UTC))//MINUTE*MINUTE))
        if merged.height:
            atomic_write(merged,path)
        remaining=list(gaps(clean(merged,a,b),a,b))
        report=root/"metadata"/symbol/(month.strftime("%Y-%m")+".json")
        report.parent.mkdir(parents=True,exist_ok=True)
        tmp=report.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"symbol":symbol,"requested":[a,b],"rows":merged.height,
            "retrieval_finished":True,"missing_ranges_ms":remaining,"note":"Retrieval finished does not imply complete coverage; no imputation"},indent=2))
        tmp.replace(report)
        print(symbol,month.date(),merged.height,"missing minutes",sum((y-x)//MINUTE for x,y in remaining),flush=True)

def aggregate(f,tf):
    f=f.sort("open_time").with_columns(pl.from_epoch("open_time",time_unit="ms").dt.replace_time_zone("UTC").alias("ts"))
    # Calendar-aligned windows, Monday start for weekly bars.
    n={"m":1,"h":60,"d":1440,"w":10080}[tf[-1]]*int(tf[:-1])
    return (f.group_by_dynamic("ts",every=tf,closed="left",label="left",start_by="monday" if tf.endswith("w") else "window")
        .agg(pl.col("open").first(),pl.col("high").max(),pl.col("low").min(),pl.col("close").last(),
             *[pl.col(c).sum() for c in ["volume","quote_volume","trades","taker_buy_base_volume","taker_buy_quote_volume"]],
             pl.len().alias("count"))
        .filter(pl.col("count")==n).drop("count")
        .with_columns(pl.col("ts").dt.epoch("ms").alias("open_time"))
        .with_columns((pl.col("open_time")+n*MINUTE-1).alias("close_time")).drop("ts").select(FIELDS))

def resample(root,symbol):
    paths=sorted((root/"futures"/symbol/"1m").glob("*.parquet"))
    # Process one month at a time, carrying a 7-day lookback for weekly boundary bars.
    tail=parse_rows([])
    for i,p in enumerate(paths):
        f=pl.read_parquet(p)
        combined=pl.concat([tail,f]).sort("open_time")
        lower=ms(date(p.stem+"-01"))
        for tf in TIMEFRAMES:
            out=aggregate(combined,tf).filter(pl.col("close_time")>=lower)
            atomic_write(out,root/"resampled"/symbol/tf/p.name)
        tail=combined.tail(10080)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=["download","resample"])
    p.add_argument("--symbols",nargs="+",default=SYMBOLS,choices=SYMBOLS)
    p.add_argument("--root",type=Path,default=Path("data"))
    p.add_argument("--start",default="2019-09-01")
    p.add_argument("--end",default=None)
    p.add_argument("--repair",action="store_true")
    a=p.parse_args()
    end=min(date(a.end) if a.end else datetime.now(UTC),datetime.now(UTC))
    end=datetime.fromtimestamp(ms(end)//MINUTE*MINUTE/1000,UTC)
    if date(a.start)>=end:
        p.error("start must precede end")
    failures=[]
    for symbol in a.symbols:
        try:
            if a.command=="download":
                download(a.root,symbol,date(a.start),end,repair=a.repair)
            else:
                resample(a.root,symbol)
        except Exception as e:
            print(symbol,"FAILED",repr(e),flush=True)
            failures.append(symbol)
    if failures:
        raise SystemExit("Failed symbols: "+", ".join(failures))

if __name__=="__main__":
    main()
