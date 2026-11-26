"""Frozen BTC incumbent confidence and calibration diagnostics, development only."""
import os
os.environ.setdefault("POLARS_MAX_THREADS","1")
import json, zipfile, hashlib
from pathlib import Path
import numpy as np
import polars as pl
from robustness_engine import A,B,ledger,pick
from execution_engine import score_trades
from experiment_store import identity

ROOT=Path("data")
def main():
    folder,=list((ROOT/"experiments").glob("6823506d*"))
    spec=json.loads((folder/"spec.json").read_text())
    manifest=json.loads((folder/"manifest.json").read_text())
    assert identity(spec)==folder.name and spec["manifest_id"]==manifest["manifest_id"]
    paths=sorted((folder/"predictions").glob("*.parquet"))
    assert {p.stem for p in paths}=={str(f["fold_id"]) for f in manifest["folds"]}
    f=pl.read_parquet(paths).sort("decision_time")
    assert f["decision_time"].n_unique()==f.height and f["symbol"].unique().to_list()==["BTCUSDT"] and f["decision_time"].min()>=A and f["label_end"].max()<B
    p=f["prediction"].to_numpy(); y=f["target_return"].to_numpy()>0
    assert np.all(np.isfinite(p)&(p>=0)&(p<=1))
    fund=ledger(ROOT); missing=fund.filter(pl.col("mark_price").is_null())["event_time"].to_numpy()
    fund=fund.filter(pl.col("mark_price").is_not_null())
    side=np.where(p>=.5,1,-1)
    t=f.with_columns(pl.Series("side",side))
    bad=np.searchsorted(missing,t["label_end"].to_numpy(),side="right")>np.searchsorted(missing,t["entry_time"].to_numpy(),side="right")
    t=t.with_columns(pl.Series("eligible",~bad))
    scored=score_trades(t.filter("eligible"),5,1,fund,{"BTCUSDT":[(A,B)]})
    rows=t.join(scored.select("decision_time","net_return"),on="decision_time",how="left",validate="1:1")
    def stats(mask):
        n=int(mask.sum())
        if not n:return {"rows":0}
        r=rows.filter(pl.Series("mask",mask))["net_return"].drop_nulls().to_numpy()
        correct=(p[mask]>=.5)==y[mask]
        return {"rows":n,"direction_accuracy":float(correct.mean()),"mean_probability":float(p[mask].mean()),"positive_rate":float(y[mask].mean()),"mean_confidence":float(np.maximum(p[mask],1-p[mask]).mean()),"brier":float(np.mean((p[mask]-y[mask])**2)),"economic_rows":len(r),"mean_net_return":float(r.mean()) if len(r) else None}
    bins=[]
    conf=np.maximum(p,1-p)
    edges=[.5,.55,.60,.65,.70,.75,.80,.90,1.000001]
    for a,b in zip(edges,edges[1:]):
        for s in (1,-1):
            bins.append({"low":a,"high":min(b,1.),"side":s,**stats((conf>=a)&(conf<b)&(side==s))})
    reliability=[{"low":i/10,"high":(i+1)/10,**stats((p>=i/10)&((p<(i+1)/10) if i<9 else (p<=1)))} for i in range(10)]
    ece=sum(b["rows"]*abs(b["mean_probability"]-b["positive_rate"]) for b in reliability if b["rows"])/len(p)
    selected=pick(f,"logistic",.7)
    actual=selected.join(scored.select("decision_time","net_return"),on="decision_time",how="left",validate="1:1")
    years=[]
    for year,g in actual.with_columns(pl.from_epoch("label_end",time_unit="ms").dt.year().alias("year")).group_by("year"):
        r=g["net_return"].drop_nulls()
        years.append({"year":int(year[0]),"trades":g.height,"known_funding":len(r),"pnl":float(r.sum())})
    out={"scope":"Exploratory frozen prediction diagnostics; confidence bins overlap in time economically; no independent inference or recalibration; selected nonoverlap ledger separately.","experiment":folder.name,"prediction_hashes":{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},"overall":stats(np.ones(len(p),dtype=bool)),"naive_brier":float(np.mean((f["baseline_prediction"].to_numpy()-y)**2)),"ece_10":ece,"confidence_bins":bins,"reliability":reliability,"selected_trades":actual.height,"selected_years":years,"selected_ledger":actual.to_dicts()}
    path=ROOT/"reports"/"11_conditional_da.zip";path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:z.writestr("metrics.json",json.dumps(out,allow_nan=False))
    tmp.replace(path);print(path)
if __name__=="__main__":main()
