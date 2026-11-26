"""Descriptive excursion diagnostics for frozen binary and cost-aware candidates."""
import os
os.environ.setdefault("POLARS_MAX_THREADS","1")
import json,zipfile,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import polars as pl
from baseline_engine import digest
from robustness_engine import A,B,ledger,pick
from funding_repair_engine import apply
from cost_aware_target_engine import select,KEYS
from execution_engine import score_trades
ROOT=Path("data")
MIN=60000
def audit(out):
    folder,=list((ROOT/"experiments").glob("6823506d*"))
    oldpaths=sorted((folder/"predictions").glob("*.parquet"))
    with zipfile.ZipFile(ROOT/"reports"/"21_cost_aware_target.zip") as zz:
        rep=json.loads(zz.read("metrics.json"))
    assert rep["status"]=="pass"
    newpaths=sorted((ROOT/"target_v2"/rep["experiment"]).glob("*.parquet"))
    assert [p.name for p in oldpaths]==[p.name for p in newpaths] and len(oldpaths)==42
    old=pl.read_parquet(oldpaths).sort("decision_time")
    new=pl.read_parquet(newpaths).sort("decision_time")
    assert old.select(KEYS).equals(new.select(KEYS))
    assert old["decision_time"].n_unique()==old.height and old["decision_time"].min()>=A and old["label_end"].max()<B
    fund=apply(ledger(ROOT)).sort("event_time")
    assert not fund["mark_price"].null_count()
    frames=[]
    for name,t,n in [("incumbent",pick(old,"logistic",.7),104),("cost_aware",select(new),63)]:
        assert t.height==n
        frames.append(score_trades(t,5,1,fund,{"BTCUSDT":[(A,B)]}).select(*KEYS,"side","net_return").with_columns(pl.lit(name).alias("candidate")))
    t=pl.concat(frames).sort("entry_time")
    raw=set(); records=[]
    for (month,),part in t.with_columns(pl.from_epoch("entry_time",time_unit="ms").dt.strftime("%Y-%m").alias("month")).group_by("month"):
        endmonth=datetime.fromtimestamp(part["label_end"].max()/1000,timezone.utc).strftime("%Y-%m")
        paths=[ROOT/"futures"/"BTCUSDT"/"1m"/(m+".parquet") for m in sorted({month,endmonth})]
        raw.update(paths)
        x=pl.read_parquet(paths,columns=["open_time","open","high","low","close"]).sort("open_time")
        ts=x["open_time"].to_numpy()
        assert len(np.unique(ts))==len(ts)
        op,hi,lo,cl=[x[k].to_numpy() for k in ("open","high","low","close")]
        for row in part.iter_rows(named=True):
            e,z=row["entry_time"],row["label_end"]
            i,j=np.searchsorted(ts,[e,z])
            assert j<len(ts) and np.array_equal(ts[i:j+1],np.arange(e,z+MIN,MIN))
            assert j-i==240 and np.isclose(op[i],row["entry_price"]) and np.isclose(op[j],row["exit_price"])
            side=row["side"];p=op[i]
            fav=side*((hi[i:j] if side==1 else lo[i:j])/p-1)
            adv=-side*((lo[i:j] if side==1 else hi[i:j])/p-1)
            assert np.isfinite(fav).all() and np.isfinite(adv).all()
            gross=side*(op[j]/p-1)
            mfe=float(max(0,gross,fav.max()));mae=float(max(0,-gross,adv.max()))
            # Gross excursions, not executable fills. Exit minute high/low excluded.
            records.append({**row,"mae":mae,"mfe":mfe,"gross_exit":float(gross),"giveback":float(mfe-gross),"minute_of_mfe":int(j-i if gross>fav.max() else fav.argmax()),"minute_of_mae":int(j-i if -gross>adv.max() else adv.argmax())})
        print(month,"done",flush=True)
    out["source_hash"]=digest(list(raw)+oldpaths+newpaths+[Path(__file__),Path("execution_engine.py"),Path("funding_repair_engine.py")])
    out["records"]=records;out["groups"]=[]
    for name in ("incumbent","cost_aware"):
        for group in ("all","wins","losses"):
            rows=[r for r in records if r["candidate"]==name and (group=="all" or (r["net_return"]>0)==(group=="wins"))]
            out["groups"].append({"candidate":name,"group":group,"n":len(rows),"quantiles":{k:np.quantile([r[k] for r in rows],[.1,.25,.5,.75,.9]).tolist() if rows else [] for k in ("mae","mfe","giveback","minute_of_mfe","minute_of_mae","net_return")}})
    out["overlapping_signal_count"]=len(set(frames[0]["decision_time"].to_list())&set(frames[1]["decision_time"].to_list()))
    out["status"]="pass"
def main():
    out={"scope":"Frozen selected trades, raw 1m OHLC excursions gross versus reference entry open; no assumed intrabar sequence or executable extrema; no TP/SL optimization, descriptive ex post only; costs in realized net return only"}
    try:audit(out)
    except Exception:out.update(status="fail",error=traceback.format_exc())
    p=ROOT/"reports"/"23_excursions.zip";tmp=p.with_suffix(".tmp")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:z.writestr("metrics.json",json.dumps(out,allow_nan=False))
    tmp.replace(p);print(p,out["status"])
if __name__=="__main__":main()
