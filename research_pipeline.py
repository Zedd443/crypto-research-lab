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

def integrity():
    """Audit frozen candidate inputs and predictions without training."""
    import zipfile, traceback
    import numpy as np
    from baseline_engine import files,load_fold,digest
    from validation_protocol import Fold,make_folds,manifest
    from experiment_store import identity
    from feature_engine import technical,BASE_FEATURES
    from cost_aware_target_engine import targets,KEYS
    from funding_repair_engine import apply
    from robustness_engine import A,B,ledger
    root=Path("data")
    out={"scope":"Frozen C01/C02 input and checkpoint audit; no model refit, no holdout, no claim of exhaustive leakage proof","folds":[],"months":[]}
    def check(ok,msg):
        if not ok:raise ValueError(msg)
    try:
        folder,=list((root/"experiments").glob("6823506d*"))
        spec=json.loads((folder/"spec.json").read_text())
        man=json.loads((folder/"manifest.json").read_text())
        check(identity(spec)==folder.name and spec["manifest_id"]==man["manifest_id"],"Identity mismatch")
        folds=make_folds(A,B,180,30)
        check(manifest(folds,B,man["holdout_end"],man["dataset_id"])==man,"Manifest mismatch")
        check(len(folds)==42 and spec["model"]=="logistic" and spec["tf"]=="1h" and spec["horizon"]==4 and spec["delay_minutes"]==15 and set(spec["features"])==set(BASE_FEATURES)-{"body"},"Specification mismatch")
        pairs=[(p,q) for p,q in files(root,["BTCUSDT"],"1h",4,15) if "2021-01"<=p.stem<="2024-12"]
        check(len(pairs)==48,"Missing feature/label partitions")
        out["current_dataset_hash"]=digest([p for pair in pairs for p in pair])
        out["original_dataset_hash"]=man["dataset_id"]
        out["dataset_hash_matches"]=out["current_dataset_hash"]==man["dataset_id"]
        with zipfile.ZipFile(root/"reports"/"21_cost_aware_target.zip") as z:rep=json.loads(z.read("metrics.json"))
        check(rep["status"]=="pass","Report 21 failed")
        fund=apply(ledger(root)).sort("event_time")
        check(not fund["mark_price"].null_count(),"Funding marks missing")
        import hashlib
        check(hashlib.sha256(fund.write_csv().encode()).hexdigest()==rep["funding_hash"],"Funding hash mismatch")
        # Rebuild hourly source from raw month, retaining only small hourly frames.
        bars=[];points=[]
        for p in sorted((root/"futures"/"BTCUSDT"/"1m").glob("*.parquet")):
            if not "2020-12"<=p.stem<="2024-12":continue
            raw=pl.read_parquet(p).sort("open_time")
            ts=raw["open_time"].to_numpy()
            check(len(ts)>0 and np.all(ts%60000==0) and np.all(np.diff(ts)==60000),"Raw minute discontinuity "+p.stem)
            bars.append(de.aggregate(raw,"1h"))
            points.append(raw.filter(pl.col("open_time")%3600000==900000).select("open_time","open"))
        hourly=pl.concat(bars).sort("open_time")
        check(hourly["open_time"].n_unique()==hourly.height,"Duplicate hourly timestamp")
        expected=technical(hourly,3600000).select("decision_time",*BASE_FEATURES)
        prices=pl.concat(points).sort("open_time")
        check(prices["open_time"].n_unique()==prices.height,"Duplicate execution timestamps")
        for p,q in pairs:
            cached=pl.read_parquet(p).filter(pl.col("decision_time")<B).select("decision_time",*BASE_FEATURES)
            merged=cached.join(expected,on="decision_time",suffix="_raw",validate="1:1")
            check(merged.height==cached.height,"Missing raw feature source "+p.stem)
            for c in BASE_FEATURES:
                a=merged[c].cast(pl.Float64).to_numpy();b=merged[c+"_raw"].cast(pl.Float64).to_numpy()
                # Compare eligible finite values; cache may have shorter warmup.
                mask=np.isfinite(a)
                check(np.allclose(a[mask],b[mask],rtol=1e-8,atol=1e-10),"Feature mismatch "+p.stem+" "+c)
            out["months"].append({"month":p.stem,"feature_rows":cached.height})
        for fold in folds:
            train,test=load_fold(pairs,spec["features"],fold)
            old=pl.read_parquet(folder/"predictions"/(str(fold.fold_id)+".parquet"))
            new=pl.read_parquet(root/"target_v2"/rep["experiment"]/(str(fold.fold_id)+".parquet"))
            check(old.select(KEYS).equals(test.select(KEYS)) and new.select(KEYS).equals(old.select(KEYS)),"Checkpoint eligibility mismatch "+str(fold.fold_id))
            check(train["label_end"].max()<fold.test_start and test["label_end"].max()<fold.test_end,"Purge violation")
            for f in (train,test):
                check(((f["entry_time"]-f["decision_time"])==900000).all() and ((f["label_end"]-f["entry_time"])==14400000).all(),"Timing mismatch")
                j=f.join(prices.rename({"open_time":"entry_time","open":"raw_entry"}),on="entry_time",validate="1:1").join(prices.rename({"open_time":"label_end","open":"raw_exit"}),on="label_end",validate="1:1")
                check(j.height==f.height and np.allclose(j["entry_price"],j["raw_entry"],rtol=1e-12,atol=0) and np.allclose(j["exit_price"],j["raw_exit"],rtol=1e-12,atol=0),"Execution price mismatch")
                check(np.allclose(j["target_return"],np.log(j["raw_exit"].to_numpy()/j["raw_entry"].to_numpy()),rtol=1e-10,atol=1e-12),"Label mismatch")
            prob=old["prediction"].to_numpy();multi=new.select("p_short","p_flat","p_long").to_numpy()
            check(np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all() and np.isfinite(multi).all() and (multi>=0).all() and np.allclose(multi.sum(1),1),"Invalid probabilities")
            check(np.allclose(old["baseline_prediction"].to_numpy(),(train["target_return"]>0).mean()),"Binary prior mismatch")
            yt=targets(train,fund)
            check(np.allclose(new.select("prior_short","prior_flat","prior_long").to_numpy(),[(yt==c).mean() for c in (-1,0,1)]),"Multiclass prior mismatch")
            check(np.array_equal(new["cost_target"].to_numpy(),targets(test,fund)),"Cost label mismatch")
            out["folds"].append({"fold":fold.fold_id,"train_rows":train.height,"test_rows":test.height})
            print("integrity fold",fold.fold_id,"pass",flush=True)
        out["status"]="pass" if out["dataset_hash_matches"] else "review_required"
    except Exception:out.update(status="fail",error=traceback.format_exc())
    p=root/"reports"/"44_research_integrity.zip";p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(".tmp")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:z.writestr("metrics.json",json.dumps(out,allow_nan=False))
    tmp.replace(p);print(p,out["status"],flush=True)
    if out["status"]!="pass":raise SystemExit(out.get("error","Dataset hash review required"))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=["discover","download","audit","compare","coverage","invalid-candles","repair-funding","repair-marks","repair-channel-marks","candidate-portfolio","time-exit","early-invalidation","technical-extended","extended-transfer","extended-funding","alt-portfolio","diversification","stateful-extended","integrity"])
    p.add_argument("--root",type=Path,default=Path("data"))
    p.add_argument("--symbols",nargs="+",choices=SYMBOLS,default=SYMBOLS)
    p.add_argument("--start",default="2019-09-01")
    p.add_argument("--end",default=datetime.now(de.UTC).strftime("%Y-%m-%d"))
    p.add_argument("--kinds",nargs="+",choices=list(aux.KINDS),default=list(aux.KINDS))
    p.add_argument("--tf",nargs="+",choices=["5m","15m","30m","1h","4h","8h","12h","1d"],default=["15m","1h","4h","1d"])
    p.add_argument("--repair",action="store_true",help="Recheck archives and rebuild resampled partitions")
    p.add_argument('--workers',type=int,choices=range(1,9),default=None,help='Concurrent pairs (download default 4; other commands default 1)')
    args=p.parse_args()
    if args.command=="integrity":
        if args.root!=Path("data"):p.error("Integrity uses default data root")
        return integrity()
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
