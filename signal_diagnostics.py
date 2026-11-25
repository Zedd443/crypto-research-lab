"""Development signal diagnostics; not portfolio backtest or final holdout."""
import argparse,json,zipfile
from pathlib import Path
import numpy as np
import polars as pl
from data_engine import write_json

def diagnose(frame,model):
    frame=frame.sort("decision_time")
    if frame["symbol"].n_unique()!=1 or frame["decision_time"].n_unique()!=frame.height:
        raise ValueError("Expected unique single-symbol predictions")
    p=frame["prediction"].to_numpy()
    actual=frame["target_return"].to_numpy()
    entry=frame["entry_price"].to_numpy()
    exit=frame["exit_price"].to_numpy()
    start=frame["entry_time"].to_numpy()
    end=frame["label_end"].to_numpy()
    if not all(np.isfinite(v).all() for v in (p,actual,entry,exit)) or np.any(entry<=0) or np.any(exit<=0) or np.any(end<=start):
        raise ValueError("Invalid prediction data")
    if model=="logistic" and (np.any(p<0) or np.any(p>1)):raise ValueError("Invalid probability")
    thresholds=[.5,.52,.55,.6,.65,.7] if model=="logistic" else [0,.0005,.001,.002,.005]
    rows=[]
    for threshold in thresholds:
        side=np.where(p>=threshold,1,np.where(p<=1-threshold,-1,0)) if model=="logistic" else np.where(p>threshold,1,np.where(p< -threshold,-1,0))
        # No overlapping trades; signal evaluated at decision time, position
        # must be closed before a new decision (conservative).
        picked=[];free=-1
        decisions=frame["decision_time"].to_numpy()
        for i in np.flatnonzero(side):
            if decisions[i]>=free:
                picked.append(i);free=end[i]
        ix=np.asarray(picked,dtype=int)
        s=side[ix]; e=entry[ix];x=exit[ix]
        gross=s*(x/e-1)
        row={"threshold":threshold,"signals":int(np.count_nonzero(side)),"trades":len(ix),
             "coverage":len(ix)/len(p),"gross_mean":float(gross.mean()) if len(ix) else None,
             "gross_win_rate":float((gross>0).mean()) if len(ix) else None,"cost_scenarios":[]}
        for fee,slip in [(2,0),(5,1),(5,3),(10,5)]:
            ef=e*(1+s*slip/10000);xf=x*(1-s*slip/10000)
            net=s*(xf-ef)/ef-fee/10000*(1+xf/ef)
            wins=net[net>0].sum();loss=-net[net<0].sum()
            row["cost_scenarios"].append({"fee_bps_per_side":fee,"slippage_bps_per_side":slip,
                "mean_after_fee_slippage":float(net.mean()) if len(ix) else None,
                "sum_fixed_notional":float(net.sum()),"win_rate":float((net>0).mean()) if len(ix) else None,
                "profit_factor":float(wins/loss) if loss else None})
        rows.append(row)
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path("data"))
    a=p.parse_args();root=a.root
    results=[];files=[]
    for path in sorted((root/"experiments").glob("*/result.json")):
        folder=path.parent
        spec=json.loads((folder/"spec.json").read_text())
        if not spec.get("prediction_export") or spec["model"] not in ("logistic","xgb"):continue
        preds=sorted((folder/"predictions").glob("*.parquet"))
        if not preds:raise ValueError("Missing predictions "+folder.name)
        f=pl.read_parquet(preds)
        result=json.loads(path.read_text())
        key=__import__("experiment_store").identity({**spec,"params":result["params"]})
        checkpoints=list((folder/"folds"/key).glob("*.json"))
        if {x.stem for x in checkpoints}!={x.stem for x in preds}:raise ValueError("Incomplete prediction export")
        for c in checkpoints:
            expected=json.loads(c.read_text())["result"]["test_rows"]
            if pl.read_parquet(folder/"predictions"/(c.stem+".parquet")).height!=expected:raise ValueError("Prediction row count mismatch")
        results.append({"id":folder.name,"spec":spec,"observations":f.height,"scenarios":diagnose(f,spec["model"])})
        files.extend([path,folder/"spec.json",folder/"manifest.json",*preds,*checkpoints])
    if not results:raise ValueError("No completed exported predictions")
    out=root/"reports"/"04_signal_diagnostics.zip";out.parent.mkdir(parents=True,exist_ok=True)
    temp=out.with_suffix(".tmp")
    with zipfile.ZipFile(temp,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("metrics.json",json.dumps({"status":"DIAGNOSTIC_ONLY","note":"Fixed threshold exploratory development results; funding excluded; fees are assumptions; no threshold selection or portfolio equity claims.","experiments":results},indent=2,allow_nan=False))
        for f in files:z.write(f,str(f.relative_to(root)))
    temp.replace(out);print(out)
if __name__=="__main__":main()
