"""Cost-aware three-class logistic target, fixed horizon and common eligibility."""
import os
os.environ.setdefault("POLARS_MAX_THREADS","1")
import json,zipfile,traceback
from pathlib import Path
import numpy as np
import polars as pl
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from baseline_engine import files,digest,load_fold
from validation_protocol import Fold
from experiment_store import identity
from robustness_engine import A,B,DAY,ledger,pick,metrics
from funding_repair_engine import apply
from execution_engine import score_trades
from data_engine import atomic_write
ROOT=Path("data")
KEYS=["symbol","decision_time","entry_time","label_end","entry_price","exit_price","target_return"]
CLASSES=np.array([-1,0,1])
def targets(f,fund):
    # Realized funding is a label component only, never an input feature.
    l=score_trades(f.select(KEYS).with_columns(pl.lit(1).alias("side")),5,1,fund,{"BTCUSDT":[(A,B)]})["net_return"].to_numpy()
    s=score_trades(f.select(KEYS).with_columns(pl.lit(-1).alias("side")),5,1,fund,{"BTCUSDT":[(A,B)]})["net_return"].to_numpy()
    assert not np.any((l>0)&(s>0))
    return np.where(l>0,1,np.where(s>0,-1,0))
def select(f):
    p=f.select("p_short","p_flat","p_long").to_numpy()
    side=np.where(p[:,2]>=.7,1,np.where(p[:,0]>=.7,-1,0))
    ix=[];free=-1
    for i,(d,e,v) in enumerate(zip(f["decision_time"],f["label_end"],side)):
        if v and d>=free:ix.append(i);free=e
    return f.with_columns(pl.Series("side",side)).with_row_index("idx").filter(pl.col("idx").is_in(ix)).drop("idx")
def audit(out):
    folder,=list((ROOT/"experiments").glob("6823506d*"))
    spec=json.loads((folder/"spec.json").read_text())
    man=json.loads((folder/"manifest.json").read_text())
    assert identity(spec)==folder.name and spec["manifest_id"]==man["manifest_id"]
    assert spec["model"]=="logistic" and spec["delay_minutes"]==15 and spec["horizon"]==4
    cols=spec["features"]
    pairs=[(p,q) for p,q in files(ROOT,["BTCUSDT"],"1h",4,15) if "2021-01"<=p.stem<="2024-12"]
    assert len(pairs)==48
    fund=apply(ledger(ROOT)).sort("event_time")
    assert not fund["mark_price"].null_count(),"Incomplete funding labels"
    params=json.loads((folder/"result.json").read_text())["params"]
    source=digest([v for pair in pairs for v in pair])
    code=digest([Path(__file__),Path("baseline_engine.py"),Path("validation_protocol.py"),Path("execution_engine.py"),Path("funding_repair_engine.py")])
    import hashlib
    funding_hash=hashlib.sha256(fund.write_csv().encode()).hexdigest()
    key=identity({"source":source,"code":code,"funding":funding_hash,"incumbent":folder.name,"params":params,"sklearn":sklearn.__version__,"polars":pl.__version__,"fee":5,"slip":1,"threshold":.7})
    cache=ROOT/"target_v2"/key
    paths=[]; oldpaths=[]; counts=[]
    for fd in man["folds"]:
        fold=Fold(**fd)
        p=cache/(str(fold.fold_id)+".parquet")
        oldpath=folder/"predictions"/p.name
        old=pl.read_parquet(oldpath)
        if not p.exists():
            train,test=load_fold(pairs,cols,fold)
            yt=targets(train,fund);yv=targets(test,fund)
            assert np.array_equal(np.unique(yt),CLASSES),"Missing training class"
            imp=SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True)
            x=imp.fit_transform(train.select(cols).to_numpy())
            v=imp.transform(test.select(cols).to_numpy())
            scale=StandardScaler()
            with threadpool_limits(limits=1):
                x=scale.fit_transform(x);v=scale.transform(v)
                model=LogisticRegression(C=params["C"],max_iter=500,random_state=spec["seed"]).fit(x,yt)
                assert np.array_equal(model.classes_,CLASSES)
                prob=model.predict_proba(v)
            prior=np.array([(yt==c).mean() for c in CLASSES])
            f=test.select(KEYS).with_columns(pl.Series("cost_target",yv),*[pl.Series(n,prob[:,i]) for i,n in enumerate(["p_short","p_flat","p_long"])],*[pl.lit(float(prior[i])).alias(n) for i,n in enumerate(["prior_short","prior_flat","prior_long"])])
            atomic_write(f,p)
        f=pl.read_parquet(p)
        assert f.select(KEYS).equals(old.select(KEYS)),"Common eligibility mismatch"
        prob=f.select("p_short","p_flat","p_long").to_numpy()
        assert np.isfinite(prob).all() and (prob>=0).all() and np.allclose(prob.sum(1),1)
        assert np.array_equal(f["cost_target"].to_numpy(),targets(f,fund)),"Cached label mismatch"
        counts.append({"fold":fold.fold_id,"classes":[int((f["cost_target"]==int(c)).sum()) for c in CLASSES]})
        paths.append(p);oldpaths.append(oldpath);print("cost aware",fold.fold_id,"done",flush=True)
    f=pl.read_parquet(paths).sort("decision_time")
    old=pl.read_parquet(oldpaths).sort("decision_time")
    assert f["decision_time"].n_unique()==f.height and f["decision_time"].min()>=A and f["label_end"].max()<B
    actual=f["cost_target"].to_numpy()
    onehot=(actual[:,None]==CLASSES).astype(float)
    prob=f.select("p_short","p_flat","p_long").to_numpy()
    prior=f.select("prior_short","prior_flat","prior_long").to_numpy()
    out.update({"experiment":key,"source":source,"code":code,"funding_hash":funding_hash,"class_order":CLASSES.tolist(),"fold_classes":counts,"class_counts":onehot.sum(0).astype(int).tolist(),"multiclass_brier":float(np.mean(np.sum((prob-onehot)**2,axis=1))),"naive_multiclass_brier":float(np.mean(np.sum((prior-onehot)**2,axis=1))),"class_accuracy":float(np.mean(CLASSES[prob.argmax(1)]==actual)),"results":[]})
    first=f["decision_time"].min()//DAY*DAY;last=(f["label_end"].max()//DAY+1)*DAY
    for name,t in [("incumbent",pick(old,"logistic",.7)),("cost_aware",select(f))]:
        rows=[]
        for fee,slip in [(5,1),(10,5)]:
            scored=score_trades(t,fee,slip,fund,{"BTCUSDT":[(A,B)]}) if t.height else t
            rows.append({"fee_bps":fee,"slippage_bps":slip,**metrics(scored,first,last)})
        out["results"].append({"name":name,"economics":rows})
    out["status"]="pass"
def main():
    out={"scope":"Three mutually exclusive realized net-profit classes: short, flat (neither profitable), long; label uses realized costs and funding only after label end; same features, chronological purge and eligibility; threshold .70 per profitable side, no tuning; fixed-notional diagnostics. Multiclass Brier not directly comparable to binary Brier."}
    try:audit(out)
    except Exception:out.update(status="fail",error=traceback.format_exc())
    p=ROOT/"reports"/"21_cost_aware_target.zip";p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(".tmp")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:z.writestr("metrics.json",json.dumps(out,allow_nan=False))
    tmp.replace(p);print(p,out["status"])
if __name__=="__main__":main()
