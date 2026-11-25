"""Benchmark identical local candle aggregation tasks without network or data writes."""
import argparse
import multiprocessing as mp
import os
import hashlib
import json
import time
import threading
import statistics
import psutil
from resource_policy import tree_rss, GIB
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS","1")

def run(task):
    import polars as pl
    import data_engine as de
    path,tf=task
    start=time.perf_counter()
    f=pl.read_parquet(path)
    if not f.height:
        raise ValueError("Empty input partition: "+str(path))
    f=de.clean(f,f["open_time"].min(),f["open_time"].max()+de.MINUTE)
    result=de.aggregate(f,tf)
    # Compare canonical rows, not compression or Parquet metadata.
    # Integer and price fields retain exact values; floating volume sums
    # are rounded only for the fingerprint, not for data storage.
    result=result.with_columns([pl.col(c).round(8) for c in result.columns
        if result.schema[c] in (pl.Float32,pl.Float64)])
    digest=hashlib.sha256(result.write_csv().encode()).hexdigest()
    return {"task":str(path)+":"+tf,"rows":result.height,"hash":digest,
            "seconds":time.perf_counter()-start}

def measure(tasks,workers):
    start=time.perf_counter()
    stop=threading.Event()
    peak=[tree_rss()]
    def monitor():
        while not stop.wait(.05):
            peak[0]=max(peak[0],tree_rss())
    thread=threading.Thread(target=monitor,daemon=True)
    thread.start()
    try:
        if workers==1:
            outputs=list(map(run,tasks))
        else:
            with mp.get_context("spawn").Pool(workers) as pool:
                outputs=pool.map(run,tasks,chunksize=1)
        peak[0]=max(peak[0],tree_rss())
    finally:
        stop.set()
        thread.join()
    return {"workers":workers,"wall_seconds":time.perf_counter()-start,
            "peak_tree_rss_bytes":peak[0],"outputs":outputs}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path("data"))
    p.add_argument("--partitions",type=int,default=12)
    p.add_argument("--workers",nargs="+",type=int,default=[1,2,4])
    p.add_argument("--repeats",type=int,default=2)
    args=p.parse_args()
    if args.partitions<1 or args.repeats<1 or not args.workers or any(w<1 or w>8 for w in args.workers):
        p.error("Invalid partition or worker count")
    paths=sorted((args.root/"futures").glob("*/1m/*.parquet"))
    if not paths:p.error("No local 1m data")
    # Evenly sample across available paths, keeping tasks identical for all runs.
    n=min(args.partitions,len(paths))
    indices=[i*(len(paths)-1)//max(1,n-1) for i in range(n)]
    tasks=[(paths[i],tf) for i in indices for tf in ("15m","1h","4h","1d")]
    runs=[]
    reference=None
    available=psutil.virtual_memory().available
    limit=max(0,available-GIB)+tree_rss()
    for w in dict.fromkeys([1,*args.workers]):
        samples=[measure(tasks,w) for _ in range(args.repeats)]
        result=samples[0]
        result["wall_seconds"]=statistics.median(s["wall_seconds"] for s in samples)
        result["peak_tree_rss_bytes"]=max(s["peak_tree_rss_bytes"] for s in samples)
        result["sample_seconds"]=[s["wall_seconds"] for s in samples]
        current=[{r["task"]:(r["rows"],r["hash"]) for r in s["outputs"]} for s in samples]
        if reference is None:reference=current[0]
        result["matches_reference"]=all(v==reference for v in current)
        result["within_memory_budget"]=result["peak_tree_rss_bytes"]<=limit
        result["speedup"]=runs[0]["wall_seconds"]/result["wall_seconds"] if runs else 1.0
        runs.append(result)
        print(w,"workers",round(result["wall_seconds"],2),"s",
              "speedup",round(result["speedup"],2),"match",result["matches_reference"],flush=True)
    valid=[r for r in runs if r["matches_reference"] and r["within_memory_budget"]]
    best=min(valid,key=lambda x:x["wall_seconds"]) if valid else None
    report={"scope":"Local independent monthly aggregation; excludes cross-month windows, ML and network; sampled process-tree RSS may double-count shared pages",
        "tasks":len(tasks),"recommended_workers_for_this_sample":best["workers"] if best else None,"available_ram_bytes":available,"runs":runs}
    out=args.root/"metadata"/"compute_benchmark.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2))
    if not all(r["matches_reference"] for r in runs):
        raise SystemExit("Parallel output differs from reference")

if __name__=="__main__":
    mp.freeze_support()
    main()
