"""Recover exact missing incumbent mark opens into an isolated overlay."""
import hashlib, io, json, zipfile, csv
from pathlib import Path
import polars as pl
import data_engine as de

ROOT=Path("data")
E,Z=1659294900000,1659309300000
STAMP="2022-07-31"
def main():
    path=ROOT/"mark_repairs"/"BTCUSDT"/"2022-07.parquet"
    url=f"{de.BASE}/daily/markPriceKlines/BTCUSDT/1m/BTCUSDT-1m-{STAMP}.zip"
    r=de.SESSION.get(url,timeout=90)
    if r.status_code==404:
        url=f"{de.BASE}/monthly/markPriceKlines/BTCUSDT/1m/BTCUSDT-1m-2022-07.zip"
        r=de.SESSION.get(url,timeout=90)
    r.raise_for_status()
    check=de.SESSION.get(url+".CHECKSUM",timeout=30);check.raise_for_status()
    sha=hashlib.sha256(r.content).hexdigest()
    if sha!=check.text.split()[0].lower():raise ValueError("Archive checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        names=[n for n in z.namelist() if n.endswith(".csv")]
        if len(names)!=1:raise ValueError("Unexpected archive members")
        rows=[row for row in csv.reader(io.StringIO(z.read(names[0]).decode("utf-8-sig"))) if row and row[0].isdigit() and E<=int(row[0])<Z]
    f=de.clean(de.parse_rows(rows),E,Z)
    if f["open_time"].to_list()!=list(range(E,Z,60000)):raise ValueError("Official archive still missing minutes")
    f=f.select(pl.col("open_time").alias("event_time"),"open")
    old=pl.read_parquet(ROOT/"mark"/"BTCUSDT"/"2022-07.parquet",columns=["event_time","open"])
    if old.filter(pl.col("event_time").is_between(E,Z,closed="left")).height:raise ValueError("Original coverage changed; audit again before repair")
    de.atomic_write(f,path)
    de.write_json(path.with_suffix(".json"),{"url":url,"archive_sha256":sha,"overlay_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"range":[E,Z],"rows":f.height,"note":"Official archive observations; original raw and feature caches unchanged"})
    print(path,f.height,"recovered")
if __name__=="__main__":main()
