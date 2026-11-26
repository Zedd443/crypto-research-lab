"""Recover channel mark gaps from checksum-verified daily archives, separate overlay."""
import csv,io,json,hashlib,zipfile
from pathlib import Path
from datetime import datetime,timezone
import polars as pl
import data_engine as de
ROOT=Path("data")
OUT=ROOT/"channel_mark_repairs"/"BTCUSDT"
MIN=60000
def main():
    with zipfile.ZipFile(ROOT/"reports"/"15_channel_portfolio.zip") as z:r=json.loads(z.read("metrics.json"))
    if r["status"]!="mark_repair_required":raise ValueError("Expected mark coverage report")
    wanted={}
    for a,b in r["gap_ranges"]:
        for t in range(a,b,MIN):
            stamp=datetime.fromtimestamp(t/1000,timezone.utc).strftime("%Y-%m-%d")
            wanted.setdefault(stamp,set()).add(t)
    for stamp,ts in sorted(wanted.items()):
        path=OUT/(stamp+".parquet");meta=path.with_suffix(".json")
        if path.exists() and meta.exists():
            m=json.loads(meta.read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest()==m["overlay_sha256"] and ts.issubset(set(pl.read_parquet(path)["event_time"].to_list())):
                print(stamp,"cached",flush=True);continue
        url=f"{de.BASE}/daily/markPriceKlines/BTCUSDT/1m/BTCUSDT-1m-{stamp}.zip"
        res=de.SESSION.get(url,timeout=90)
        if res.status_code==404:
            url=f"{de.BASE}/monthly/markPriceKlines/BTCUSDT/1m/BTCUSDT-1m-{stamp[:7]}.zip"
            res=de.SESSION.get(url,timeout=90)
        if res.status_code==404:
            print(stamp,"unavailable",flush=True);continue
        res.raise_for_status()
        ck=de.SESSION.get(url+".CHECKSUM",timeout=30);ck.raise_for_status()
        sha=hashlib.sha256(res.content).hexdigest()
        if sha!=ck.text.split()[0].lower():raise ValueError("Archive checksum mismatch")
        with zipfile.ZipFile(io.BytesIO(res.content)) as z:
            names=[n for n in z.namelist() if n.endswith(".csv")]
            if len(names)!=1:raise ValueError("Unexpected archive members")
            rows=[row for row in csv.reader(io.StringIO(z.read(names[0]).decode("utf-8-sig"))) if row and row[0].isdigit() and int(row[0]) in ts]
        f=de.clean(de.parse_rows(rows),min(ts),max(ts)+MIN).select(pl.col("open_time").alias("event_time"),"open")
        if path.exists() and meta.exists():
            m=json.loads(meta.read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest()!=m["overlay_sha256"]:raise ValueError("Existing overlay hash mismatch")
            f=pl.concat([pl.read_parquet(path),f]).unique().sort("event_time")
            if f["event_time"].n_unique()!=f.height:raise ValueError("Conflicting repair")
        de.atomic_write(f,path)
        de.write_json(meta,{"url":url,"archive_sha256":sha,"overlay_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"rows":f.height})
        print(stamp,len(ts.intersection(f["event_time"].to_list())),"/",len(ts),flush=True)
if __name__=="__main__":main()
