"""Read-only BTC 15m-delay candidate integrity audit; no model fit or holdout."""
import os
os.environ.setdefault("POLARS_MAX_THREADS","1")
import json, zipfile, traceback
from pathlib import Path
import numpy as np
import polars as pl
from baseline_engine import files, digest, load_fold
from validation_protocol import Fold, manifest
from experiment_store import identity
from robustness_engine import A, B, ledger, pick
from execution_engine import score_trades

IDS = ["6823506d840bd9119d0550bb8ab77eb37ef7d3bf2faa3850f19d8fa5c3f29814",
"2c393ded7addc2e024ca70fb8feb24aaa2b2c87151b69e7b1e3ad31100fe7b5c",
"bd5df1b45db8890007e318fc0698c334cc060a8ff911d1b64138873ab0ed4dd9",
"e943ad3fac825c8e8247b607d03b6666496b39a1f0985d9f1e9bbb4b539fdce8"]
KEYS=["symbol","decision_time","entry_time","label_end","entry_price","exit_price","target_return"]
def require(ok,msg):
    if not ok: raise ValueError(msg)

def audit(root,out):
    pairs=[(p,q) for p,q in files(root,["BTCUSDT"],"1h",4,15) if "2021-01"<=p.stem<="2024-12"]
    require(len(pairs)==48,"Expected 48 paired development months")
    source=digest([p for pair in pairs for p in pair])
    out["dataset_hash"]=source
    # Only read development minute prices, one month at a time.
    raw=sorted((root/"futures"/"BTCUSDT"/"1m").glob("*.parquet"))
    raw=[p for p in raw if "2021-01"<=p.stem<="2024-12"]
    require(len(raw)==48,"Expected 48 raw months")
    labels=pl.read_parquet([q for p,q in pairs]).sort("decision_time")
    require(labels["decision_time"].n_unique()==labels.height,"Duplicate labels")
    require(labels["symbol"].unique().to_list()==["BTCUSDT"],"Wrong label symbol")
    require(labels.filter((pl.col("decision_time")<A)|(pl.col("label_end")>=B)|(pl.col("entry_time")-pl.col("decision_time")!=900000)|(pl.col("label_end")-pl.col("entry_time")!=14400000)).height==0,"Label timing")
    require(np.allclose(labels["target_return"].to_numpy(),np.log(labels["exit_price"].to_numpy()/labels["entry_price"].to_numpy()),rtol=1e-9,atol=1e-12),"Label arithmetic")
    endpoints=pl.concat([labels.select(pl.col(t).alias("open_time"),pl.col(p).alias("expected")) for t,p in [("entry_time","entry_price"),("label_end","exit_price")]])
    matched=0; prev=A-60000
    for p in raw:
        x=pl.read_parquet(p,columns=["open_time","open"]).sort("open_time")
        t=x["open_time"].to_numpy()
        require(len(t)>0 and t[0]==prev+60000 and np.all(np.diff(t)==60000),"Raw minute gap or duplicate "+p.stem)
        require(np.all(np.isfinite(x["open"].to_numpy()) & (x["open"].to_numpy()>0)),"Invalid raw open")
        prev=int(t[-1])
        e=endpoints.filter(pl.col("open_time").is_between(int(t[0]),prev)).join(x,on="open_time",how="left",validate="m:1")
        require(np.allclose(e["expected"].to_numpy(),e["open"].to_numpy(),rtol=1e-10,atol=1e-10),"Raw endpoint mismatch "+p.stem)
        matched+=e.height
    require(prev==B-60000 and matched==endpoints.height,"Raw coverage or unmatched endpoints")
    out["raw_endpoint_checks"]=matched
    fund=ledger(root)
    missing=fund.filter(pl.col("mark_price").is_null())["event_time"].to_numpy()
    fund=fund.filter(pl.col("mark_price").is_not_null())
    common={}
    for idx,eid in enumerate(IDS):
        folder=root/"experiments"/eid
        s=json.loads((folder/"spec.json").read_text()); m=json.loads((folder/"manifest.json").read_text())
        require(identity(s)==eid,"Experiment identity mismatch")
        folds=[Fold(**f) for f in m["folds"]]
        require(m==manifest(folds,m["holdout_start"],m["holdout_end"],source) and s["manifest_id"]==m["manifest_id"] and m["holdout_start"]==B,"Manifest mismatch")
        params=json.loads((folder/"result.json").read_text())["params"]
        key=identity({**s,"params":params})
        expected={str(f.fold_id) for f in folds}
        require({p.stem for p in (folder/"predictions").glob("*.parquet")}==expected,"Prediction fold coverage")
        require({p.stem for p in (folder/"folds"/key).glob("*.json")}==expected,"Checkpoint fold coverage")
        rows=[]; frames=[]
        for f in folds:
            tr,te=load_fold(pairs,s["features"],f,s.get("missing_policy","median"))
            p=pl.read_parquet(folder/"predictions"/(str(f.fold_id)+".parquet")).sort("decision_time")
            rec=json.loads((folder/"folds"/key/(str(f.fold_id)+".json")).read_text())
            r=rec["result"]
            require(rec["identity"]==key and r["train_rows"]==tr.height and r["test_rows"]==te.height==p.height,"Checkpoint counts")
            require(p.select(KEYS).equals(te.select(KEYS)),"Export keys/labels mismatch")
            if idx==2: common[f.fold_id]=(tr["decision_time"].to_list(),te["decision_time"].to_list())
            if idx==3: require(common[f.fold_id]==(tr["decision_time"].to_list(),te["decision_time"].to_list()),"Common eligibility mismatch")
            y=p["target_return"].to_numpy()>0
            prob=p["prediction"].to_numpy(); base=p["baseline_prediction"].to_numpy()
            require(np.all(np.isfinite(prob)&(prob>=0)&(prob<=1)),"Invalid probability")
            require(np.allclose(base,(tr["target_return"].to_numpy()>0).mean()),"Naive baseline mismatch")
            loss=float(np.mean((prob-y)**2)); bl=float(np.mean((base-y)**2))
            require(np.isclose(loss,r["loss"],rtol=1e-9) and np.isclose(bl,r["baseline_loss"],rtol=1e-9),"Brier checkpoint mismatch")
            rows.append({"fold":f.fold_id,"train":tr.height,"test":p.height,"brier":loss,"baseline_brier":bl})
            frames.append(p)
        p=pl.concat(frames).sort("decision_time")
        t=pick(p,"logistic",.7)
        affected=np.searchsorted(missing,t["label_end"].to_numpy(),side="right")>np.searchsorted(missing,t["entry_time"].to_numpy(),side="right")
        t=t.filter(pl.Series("eligible",~affected))
        scored=score_trades(t,5,1,fund,{"BTCUSDT":[(A,B)]})
        y=p["target_return"].to_numpy()>0; prob=p["prediction"].to_numpy(); base=p["baseline_prediction"].to_numpy()
        out["experiments"].append({"id":eid,"folds":rows,"rows":p.height,"brier":float(np.mean((prob-y)**2)),"baseline_brier":float(np.mean((base-y)**2)),"direction_accuracy":float(np.mean((prob>=.5)==y)),"trades":scored.height,"pnl":float(scored["net_return"].sum()),"expectancy":float(scored["net_return"].mean()),"excluded_funding":int(affected.sum())})
        print(eid[:8],"verified",flush=True)

def main():
    root=Path("data"); out={"scope":"development only; current data and export consistency, not model replay or full feature causality certification","experiments":[]}
    try:
        audit(root,out);out["status"]="pass"
    except Exception:
        out["status"]="fail";out["error"]=traceback.format_exc()
    path=root/"reports"/"08_integrity.zip";path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:z.writestr("metrics.json",json.dumps(out,indent=2,allow_nan=False))
    tmp.replace(path);print(path,out["status"],flush=True)
    if out["status"]!="pass":raise SystemExit(out["error"])
if __name__=="__main__":main()
