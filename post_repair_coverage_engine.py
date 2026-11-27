"""Report 34: local development coverage audit, not listing verification."""
import argparse, json, zipfile, csv, io
from pathlib import Path
from collections import Counter
import polars as pl
from data_engine import date, ms, months, MINUTE, clean, gaps
from derivatives_engine import KINDS, METRICS
from universe import CRYPTO_SYMBOLS

def audit(root):
    rows=[]; errors=[]
    for symbol in CRYPTO_SYMBOLS:
        for start,end in months(date("2021-01-01"),date("2025-01-01")):
            month=start.strftime("%Y-%m"); a,b=ms(start),ms(end)
            p=root/"futures"/symbol/"1m"/(month+".parquet")
            raw={"symbol":symbol,"month":month,"kind":"futures","rows":0,"status":"NO_RAW_UNVERIFIED"}
            try:
                if p.exists():
                    f=clean(pl.read_parquet(p),a,b)
                    raw["rows"]=f.height
                    if f.height:
                        lo,hi=int(f["open_time"].min()),int(f["open_time"].max())+MINUTE
                        raw.update(first=lo,last_exclusive=hi,internal_missing=sum((y-x)//MINUTE for x,y in gaps(f,lo,hi)),boundary_missing=((lo-a)+(b-hi))//MINUTE)
                        raw["status"]="FULL_MONTH" if f.height==(b-a)//MINUTE else "PARTIAL_UNVERIFIED"
                    del f
            except Exception as e:
                raw["status"]="ERROR";errors.append(dict(raw,error=repr(e)))
            rows.append(raw)
            if not raw.get("first"): continue
            lo,hi=raw["first"],raw["last_exclusive"]
            for kind in KINDS:
                p=root/kind/symbol/(month+".parquet")
                rec={"symbol":symbol,"month":month,"kind":kind,"rows":0,"status":"ABSENT"}
                try:
                    marker=p.with_suffix(".json")
                    if marker.exists():
                        meta=json.loads(marker.read_text(encoding="utf-8"))
                        rec.update(marker_rows=meta.get("rows"),marker_complete=meta.get("complete"),retrieval_finished=meta.get("retrieval_finished"))
                    if p.exists():
                        f=pl.read_parquet(p).filter(pl.col("event_time").is_between(lo,hi,closed="left")).sort("event_time")
                        t=f["event_time"];rec["rows"]=f.height
                        rec["duplicates"]=f.height-t.n_unique()
                        cols=METRICS if kind=="metrics" else (["funding_rate"] if kind=="funding" else ["open","high","low","close"])
                        rec["invalid_values"]={c:f.filter(pl.col(c).is_null()|~pl.col(c).is_finite()).height for c in cols}
                        if kind=="funding":
                            rec["observed_intervals_ms"]={str(k):v for k,v in Counter(t.diff().drop_nulls().to_list()).items()}
                            rec["status"]="SCHEDULE_UNVERIFIED" if f.height else "EMPTY"
                        else:
                            step=300000 if kind=="metrics" else MINUTE
                            rec["expected"]=(hi+step-1)//step-(lo+step-1)//step
                            rec["off_grid"]=f.filter(pl.col("event_time")%step!=0).height
                            rec["missing_grid"]=rec["expected"]-f.filter(pl.col("event_time")%step==0)["event_time"].n_unique()
                            rec["status"]="COMPLETE_GRID" if rec["missing_grid"]==0 and rec["off_grid"]==0 else "PARTIAL"
                        if rec["duplicates"] or any(rec["invalid_values"].values()):rec["status"]="INVALID"
                        del f,t
                except Exception as e:
                    rec["status"]="ERROR";errors.append(dict(rec,error=repr(e)))
                rows.append(rec)
        print(symbol,"audited",flush=True)
    return rows,errors

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path("data"));a=p.parse_args()
    rows,errors=audit(a.root)
    summary={s:{k:dict(Counter(r["status"] for r in rows if r["symbol"]==s and r["kind"]==k)) for k in ["futures",*KINDS]} for s in CRYPTO_SYMBOLS}
    queue=[r for r in rows if r["kind"]!="futures" and r["status"] in ("ABSENT","EMPTY","PARTIAL","INVALID","ERROR")]
    out=a.root/"reports"/"34_post_repair_coverage.zip";out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix(".tmp")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:
        for name,obj in [("summary.json",summary),("partitions.json",rows),("review_queue.json",queue),("errors.json",errors),("status.json",{"status":"ERROR" if errors else "PASS","scope":"2021-2024 observed raw bounds only; listing, funding schedules and publication time unverified","partitions":len(rows),"review_items":len(queue)})]:
            z.writestr(name,json.dumps(obj,indent=2))
    tmp.replace(out);print(out, "review items",len(queue),"errors",len(errors))
    if errors:raise SystemExit(1)
if __name__=="__main__":main()
