"""Discover actual archive months before downloading; audit without imputing data."""
import argparse
import os
import multiprocessing as mp
import time

# Each spawned worker has its own HTTP session and bounded Polars thread pool.
os.environ.setdefault('POLARS_MAX_THREADS', '2')
import json
import re
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timedelta
import polars as pl
import data_engine as de
import derivatives_engine as aux
from universe import SYMBOLS, TRADFI_SYMBOLS, TRADFI_START

S3="https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"

def inventory(symbol):
    prefix=f"data/futures/um/monthly/klines/{symbol}/1m/"
    keys=[]
    token=None
    while True:
        params={"list-type":"2","prefix":prefix}
        if token:params["continuation-token"]=token
        r=de.SESSION.get(S3,params=params,timeout=60)
        r.raise_for_status()
        xml=ET.fromstring(r.content)
        keys.extend(e.text for e in xml.findall(".//{*}Contents/{*}Key"))
        if xml.findtext("{*}IsTruncated")!="true":break
        token=xml.findtext("{*}NextContinuationToken")
        if not token:raise ValueError("Missing pagination token")
    pattern=re.compile(re.escape(symbol)+r"-1m-(\d{4}-\d{2})\.zip$")
    return sorted({m.group(1) for k in keys if (m:=pattern.search(k))})

def save_json(path,obj):
    de.write_json(path,obj)

def discover(root,symbol):
    stamps=inventory(symbol)
    result={"symbol":symbol,"archive_months":stamps,"first_archive_month":stamps[0] if stamps else None,
            "last_archive_month":stamps[-1] if stamps else None,
            "note":"Monthly archive coverage, not verified contract listing or delisting dates"}
    save_json(root/"metadata"/symbol/"archive_inventory.json",result)
    print(symbol,len(stamps),"archive months",flush=True)
    return stamps

def native_compare(root,symbol,tf,stamp):
    url=f"{de.BASE}/monthly/klines/{symbol}/{tf}/{symbol}-{tf}-{stamp}.zip"
    rows=aux.read_zip(url)
    if rows is None:return {"status":"native_archive_missing"}
    native=de.parse_rows([r for r in rows if r and r[0].isdigit()])
    p=root/"resampled"/symbol/tf/(stamp+".parquet")
    if not p.exists():return {"status":"resampled_missing"}
    actual=pl.read_parquet(p)
    if actual["open_time"].n_unique()!=actual.height or native["open_time"].n_unique()!=native.height:
        raise ValueError("Duplicate native or resampled candle timestamps")
    joined=actual.join(native,on="open_time",suffix="_native")
    mismatches={}
    for c in de.FIELDS:
        if c=="open_time":continue
        x=pl.col(c);y=pl.col(c+"_native")
        tol=0 if c in ("trades","close_time") else 1e-8
        mismatches[c]=joined.filter(x.is_null() | y.is_null() | ~x.is_finite() | ~y.is_finite() | ((x-y).abs()>tol*(1+y.abs()))).height
    missing=native.join(actual.select("open_time"),on="open_time",how="anti").height
    extra=actual.join(native.select("open_time"),on="open_time",how="anti").height
    passed=bool(joined.height and not missing and not extra and not any(mismatches.values()))
    return {"status":"passed" if passed else "mismatch","matched_rows":joined.height,"native_rows":native.height,
            "resampled_rows":actual.height,"missing_native_timestamps":missing,"extra_resampled_timestamps":extra,"mismatches":mismatches}

def audit(root,symbol):
    aux.audit(root,symbol)
    report=[]
    for p,a,b in aux.coverage(root,symbol):
        base=pl.read_parquet(p)
        row={"month":p.stem,"ohlcv_rows":base.height}
        for kind in aux.KINDS:
            path=root/kind/symbol/p.name
            if not path.exists():
                row[kind]={"status":"missing"};continue
            f=pl.read_parquet(path).filter(pl.col("event_time").is_between(a,b,closed="left")).sort("event_time")
            nulls={c:f[c].null_count() for c in f.columns}
            info={"rows":f.height,"duplicates":f.height-f["event_time"].n_unique(),"nulls":nulls}
            if kind!="funding":
                step=300000 if kind=="metrics" else de.MINUTE
                expected=base.select((pl.col("open_time")//step*step).alias("event_time")).unique()
                # A partial 5m bin at the beginning is not an expected observation.
                expected=expected.filter(pl.col("event_time")>=a)
                info["expected_grid_rows"]=expected.height
                info["missing_grid_rows"]=expected.join(f.select("event_time"),on="event_time",how="anti").height
            else:
                info["interval_minutes_observed"]=(f["event_time"].diff().drop_nulls().unique().sort() / 60000).to_list()
            row[kind]=info
        report.append(row)
    save_json(root/"metadata"/symbol/"cross_dataset_audit.json",report)
    print(symbol,"audited",len(report),"months",flush=True)

def run_symbol(task):
    args,symbol,start,end=task
    print(symbol,"START",flush=True)
    try:
        if args.command=="discover":discover(args.root,symbol)
        elif args.command=="audit":audit(args.root,symbol)
        elif args.command=="compare":
            report={p.stem:{tf:native_compare(args.root,symbol,tf,p.stem) for tf in args.tf}
                    for p,a,b in aux.coverage(args.root,symbol) if a<de.ms(end) and b>de.ms(start)}
            save_json(args.root/"metadata"/symbol/"native_comparison.json",report)
            print(symbol,json.dumps(report),flush=True)
            if not report or any(v["status"]!="passed" for month in report.values() for v in month.values()):
                raise ValueError("Native comparison incomplete or mismatched; inspect report")
        else:
            stamps=discover(args.root,symbol)
            # Skip only months before first and after last known archive when the
            # contract is confirmed delisted; active or unknown status stays open.
            from universe import UNIVERSE
            if not stamps and symbol not in TRADFI_SYMBOLS:raise ValueError("No monthly archive found; inspect symbol manually")
            first=de.date(TRADFI_START[symbol]) if symbol in TRADFI_SYMBOLS else de.date(stamps[0]+"-01")
            last=de.date(stamps[-1]+"-01") if stamps else first
            bound_end=(last+timedelta(days=32)).replace(day=1)
            # Include a trailing month for possible final partial daily archive.
            if symbol in UNIVERSE["delisted"]:
                bound_end=(bound_end+timedelta(days=32)).replace(day=1)
                end_symbol=min(end,bound_end)
            else:end_symbol=end
            start_symbol=max(start,first)
            if start_symbol>=end_symbol:return symbol,None
            errors={}
            def attempt(stage,fn):
                try:fn()
                except Exception as exc:
                    errors[stage]=repr(exc)
                    print(symbol,stage,"FAILED",repr(exc),flush=True)
            attempt("ohlcv",lambda:de.download(args.root,symbol,start_symbol,end_symbol,repair=args.repair))
            attempt("resample",lambda:de.resample(args.root,symbol,force=args.repair))
            for kind in args.kinds:
                if symbol in TRADFI_SYMBOLS and kind=="spot":continue
                attempt(kind,lambda:aux.download(args.root,symbol,kind,start_symbol,end_symbol,repair=args.repair))
            attempt("audit",lambda:audit(args.root,symbol))
            save_json(args.root/"metadata"/symbol/"repair_queue.json",{"symbol":symbol,"start":start_symbol.isoformat(),"end":end_symbol.isoformat(),"errors":errors})
            if errors:return symbol,json.dumps(errors)
        return symbol,None
    except Exception as e:
        print(symbol,"FAILED",repr(e),flush=True)
        return symbol,repr(e)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=["discover","download","audit","compare","coverage","invalid-candles","repair-funding","repair-marks","repair-channel-marks","candidate-portfolio","time-exit","early-invalidation","technical-extended","extended-transfer","extended-funding","alt-portfolio","diversification","stateful-extended"])
    p.add_argument("--root",type=Path,default=Path("data"))
    p.add_argument("--symbols",nargs="+",choices=SYMBOLS,default=SYMBOLS)
    p.add_argument("--start",default="2019-09-01")
    p.add_argument("--end",default=datetime.now(de.UTC).strftime("%Y-%m-%d"))
    p.add_argument("--kinds",nargs="+",choices=list(aux.KINDS),default=list(aux.KINDS))
    p.add_argument("--tf",nargs="+",choices=["5m","15m","30m","1h","4h","8h","12h","1d"],default=["15m","1h","4h","1d"])
    p.add_argument("--repair",action="store_true",help="Recheck archives and rebuild resampled partitions")
    p.add_argument('--workers',type=int,choices=range(1,9),default=None,help='Concurrent pairs (download default 4; other commands default 1)')
    args=p.parse_args()
    # Stable entry point; keep historical modules intact for experiment hashes.
    import subprocess, sys
    commands={
        "coverage": ["post_repair_coverage_engine.py","--root",str(args.root)],
        "invalid-candles": ["invalid_candle_audit.py"],
        "repair-funding": ["funding_repair_engine.py"],
        "repair-marks": ["mark_repair_engine.py"],
        "repair-channel-marks": ["channel_mark_repair_engine.py"],
        "candidate-portfolio": ["portfolio_engine.py","--candidate","compare"],
        "time-exit": ["portfolio_engine.py","--candidate","exit"],
        "early-invalidation": ["portfolio_engine.py","--candidate","invalidation"],
        "technical-extended": ["technical_rule_engine.py","--extended"],
        "extended-transfer": ["multi_asset_transfer_engine.py","--extended"],
        "extended-funding": ["multi_asset_funding_engine.py","--extended"],
        "alt-portfolio": ["portfolio_engine.py","--candidate","alt"],
        "diversification": ["portfolio_engine.py","--candidate","diversification"],
        "stateful-extended": ["technical_rule_engine.py","--stateful"],
    }
    if args.command in commands:
        if args.command!="coverage" and args.root!=Path("data"):
            p.error("This historical engine requires the default data root")
        raise SystemExit(subprocess.call([sys.executable,*commands[args.command]]))
    start,end=de.date(args.start),de.date(args.end)
    if start>=end:p.error("start must precede end")
    symbols=list(dict.fromkeys(args.symbols))
    tasks=[(args,symbol,start,end) for symbol in symbols]
    workers=min(args.workers if args.workers is not None else (4 if args.command=="download" else 1),len(tasks))
    failures=[]
    begun=time.monotonic()
    print(f"Running {len(tasks)} unique pairs with {workers} worker(s)",flush=True)
    def consume(results):
        for done,(symbol,error) in enumerate(results,1):
            if error:failures.append(symbol)
            elapsed=(time.monotonic()-begun)/60
            print(f"[{done}/{len(tasks)}] {symbol} {'FAILED' if error else 'DONE'} | elapsed {elapsed:.1f} min",flush=True)
    if workers==1:
        consume(map(run_symbol,tasks))
    else:
        # Spawn is explicit for consistent Windows/Linux behavior. On Ctrl+C the
        # pool context terminates workers; finished atomic Parquet files survive.
        with mp.get_context("spawn").Pool(workers) as pool:
            consume(pool.imap_unordered(run_symbol,tasks,chunksize=1))
    if failures:raise SystemExit("Failed: "+", ".join(failures))
if __name__=="__main__":
    mp.freeze_support()
    main()
