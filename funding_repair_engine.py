"""Recover missing funding minute-open marks; immutable raw, independent overlay."""
import csv,io,json,hashlib,zipfile
from datetime import datetime,timezone
from pathlib import Path
import polars as pl
import data_engine as de
from robustness_engine import ledger

ROOT=Path("data")
OUT=ROOT/"funding_repairs"/"BTCUSDT.json"
def main():
    f=ledger(ROOT)
    missing=f.filter(pl.col("mark_price").is_null())["event_time"].to_list()
    existing=json.loads(OUT.read_text()) if OUT.exists() else {}
    for stamp in sorted({datetime.fromtimestamp(t/1000,timezone.utc).strftime("%Y-%m-%d") for t in missing if str(t) not in existing}):
        wanted=[t for t in missing if datetime.fromtimestamp(t/1000,timezone.utc).strftime("%Y-%m-%d")==stamp and str(t) not in existing]
        url=f"{de.BASE}/daily/markPriceKlines/BTCUSDT/1m/BTCUSDT-1m-{stamp}.zip"
        r=de.SESSION.get(url,timeout=90)
        if r.status_code==404:
            url=f"{de.BASE}/monthly/markPriceKlines/BTCUSDT/1m/BTCUSDT-1m-{stamp[:7]}.zip"
            r=de.SESSION.get(url,timeout=90)
        if r.status_code==404:
            print(stamp,"archive unavailable",flush=True);continue
        r.raise_for_status()
        check=de.SESSION.get(url+".CHECKSUM",timeout=30);check.raise_for_status()
        sha=hashlib.sha256(r.content).hexdigest()
        if sha!=check.text.split()[0].lower():raise ValueError("Checksum mismatch")
        minutes={t//60000*60000 for t in wanted}
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            names=[n for n in z.namelist() if n.endswith(".csv")]
            if len(names)!=1:raise ValueError("Unexpected CSV count")
            rows=[row for row in csv.reader(io.StringIO(z.read(names[0]).decode("utf-8-sig"))) if row and row[0].isdigit() and int(row[0]) in minutes]
        if rows:
            candles=de.clean(de.parse_rows(rows),min(minutes),max(minutes)+60000)
            prices=dict(candles.select("open_time","open").iter_rows())
            for t in wanted:
                m=t//60000*60000
                if m in prices:existing[str(t)]={"minute":m,"mark_price":prices[m],"url":url,"archive_sha256":sha}
        de.write_json(OUT,existing)
        print(stamp,"recovered",sum(str(t) in existing for t in wanted),"/",len(wanted),flush=True)
    print("Recovered",sum(str(t) in existing for t in missing),"/",len(missing),flush=True)

def apply(f):
    if not OUT.exists():return f
    import math
    records=json.loads(OUT.read_text())
    rows=[]
    for t,v in records.items():
        t=int(t);p=float(v["mark_price"])
        if v["minute"]!=t//60000*60000 or not math.isfinite(p) or p<=0 or not v["url"].startswith(de.BASE+"/") or len(v["archive_sha256"])!=64:raise ValueError("Invalid funding repair provenance")
        rows.append((t,p))
    if not rows:return f
    overlay=pl.DataFrame(rows,schema={"event_time":pl.Int64,"recovered":pl.Float64},orient="row")
    f=f.join(overlay,on="event_time",how="left",validate="1:1")
    if f.filter(pl.col("mark_price").is_not_null()&pl.col("recovered").is_not_null()&(pl.col("mark_price")!=pl.col("recovered"))).height:raise ValueError("Funding repair conflicts with original")
    return f.with_columns(pl.coalesce("mark_price","recovered").alias("mark_price")).drop("recovered")
if __name__=="__main__":main()
