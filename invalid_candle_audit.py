"""Read-only invalid candle diagnosis; no raw writes."""
import json,zipfile,traceback
from pathlib import Path
from datetime import datetime,timezone,timedelta
import data_engine as d
import polars as pl
ROOT=Path("data")
def diagnose(symbol):
    a=d.ms(d.date("2023-11-01"));b=d.ms(d.date("2023-12-01"))
    f=d.archive(symbol,"monthly","2023-11")
    assert f is not None
    checks={"nonfinite":pl.any_horizontal([pl.col(c).is_null()|~pl.col(c).is_finite() for c in d.FIELDS]),"open_time":pl.col("open_time")%60000!=0,"close_time":pl.col("close_time")!=pl.col("open_time")+59999,"price":pl.min_horizontal("open","high","low","close")<=0,"high":pl.col("high")<pl.max_horizontal("open","low","close"),"low":pl.col("low")>pl.min_horizontal("open","high","close"),"negative":pl.min_horizontal("volume","quote_volume","trades","taker_buy_base_volume","taker_buy_quote_volume")<0,"taker_base":pl.col("taker_buy_base_volume")>pl.col("volume")+1e-8,"taker_quote":pl.col("taker_buy_quote_volume")>pl.col("quote_volume")+1e-8}
    x=f.filter(pl.col("open_time").is_between(a,b,closed="left")).with_columns(*[v.alias("bad_"+k) for k,v in checks.items()])
    bad=x.filter(pl.any_horizontal(["bad_"+k for k in checks]))
    out={"symbol":symbol,"monthly_rows":f.height,"bad_rows":[]}
    for row in bad.iter_rows(named=True):
        t=row["open_time"];day=datetime.fromtimestamp(t/1000,timezone.utc).strftime("%Y-%m-%d")
        record={"monthly":row}
        for name,fn in [("daily",lambda:d.archive(symbol,"daily",day)),("rest",lambda:d.rest(symbol,t,t+60000))]:
            try:
                q=fn()
                record[name]=[] if q is None else q.filter(pl.col("open_time")==t).to_dicts()
            except Exception as e:record[name+"_error"]=repr(e)
        out["bad_rows"].append(record)
    return out
def main():
    out={"results":[]}
    for s in ("DOTUSDT","SUIUSDT","XRPUSDT"):
        try:out["results"].append(diagnose(s))
        except Exception:out["results"].append({"symbol":s,"error":traceback.format_exc()})
    out["status"]="pass" if all("error" not in r for r in out["results"]) else "partial"
    p=ROOT/"reports"/"33_invalid_candles.zip";p.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(p.with_suffix(".tmp"),"w",zipfile.ZIP_DEFLATED) as z:z.writestr("metrics.json",json.dumps(out,indent=2,default=str))
    p.with_suffix(".tmp").replace(p);print(p,out["status"])
if __name__=="__main__":main()
