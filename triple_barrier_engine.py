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
MIN=60000
def build_paths(f,raw):
    # Monthly inputs plus next month for boundary-crossing paths.
    from datetime import datetime,timezone
    result=[]
    for month,part in f.with_columns(pl.from_epoch("entry_time",time_unit="ms").dt.strftime("%Y-%m").alias("month")).group_by("month"):
        month=month[0]
        paths=[p for p in raw if p.stem==month or p.stem==datetime.fromtimestamp((int(part["label_end"].max()))/1000,timezone.utc).strftime("%Y-%m")]
        x=pl.read_parquet(paths,columns=["open_time","open","close"]).sort("open_time")
        ts=x["open_time"].to_numpy(); op=x["open"].to_numpy();cl=x["close"].to_numpy()
        assert len(np.unique(ts))==len(ts)
        for row in part.iter_rows(named=True):
            e,z=row["entry_time"],row["label_end"]
            i=np.searchsorted(ts,e);j=np.searchsorted(ts,z)
            if j>=len(ts) or not np.array_equal(ts[i:j+1],np.arange(e,z+MIN,MIN)):raise ValueError(f"Missing minute path {e}")
            assert np.isclose(op[i],row["entry_price"]) and np.isclose(op[j],row["exit_price"])
            v=row["volatility_20"]
            assert v is not None and np.isfinite(v) and v>=0
            width=max(.0012,2*v)
            ret=np.log(cl[i:j]/op[i])
            hits=np.flatnonzero(np.abs(ret)>=width)
            k=i+int(hits[0])+1 if len(hits) else j
            result.append({"symbol":row["symbol"],"decision_time":row["decision_time"],"barrier_end":int(ts[k]),"barrier_exit":float(op[k]),"barrier_width":float(width),"barrier_hit":int(np.sign(ret[hits[0]])) if len(hits) else 0})
    return pl.DataFrame(result).sort("decision_time")
def adapt(f,paths):
    return f.join(paths,on=["symbol","decision_time"],how="left",validate="1:1").with_columns(pl.col("label_end").alias("fixed_end"),pl.col("exit_price").alias("fixed_exit")).with_columns(pl.col("barrier_end").alias("label_end"),pl.col("barrier_exit").alias("exit_price"))
def targets(f,fund):
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
    raw=[p for p in sorted((ROOT/"futures"/"BTCUSDT"/"1m").glob("*.parquet")) if "2021-01"<=p.stem<="2024-12"]
    source=digest([v for pair in pairs for v in pair]+raw)
    code=digest([Path(__file__),Path("baseline_engine.py"),Path("validation_protocol.py"),Path("execution_engine.py"),Path("funding_repair_engine.py")])
    import hashlib
    funding_hash=hashlib.sha256(fund.write_csv().encode()).hexdigest()
    key=identity({"source":source,"code":code,"funding":funding_hash,"incumbent":folder.name,"params":params,"sklearn":sklearn.__version__,"polars":pl.__version__,"fee":5,"slip":1,"threshold":.7})
    cache=ROOT/"target_v2"/key
    labelpath=cache/"barrier_paths.parquet"
    if not labelpath.exists():
        # Build on the same original eligible rows; only development labels.
        frames=[]
        for p,q in pairs:
            x=pl.read_parquet(p).filter(pl.all_horizontal([pl.col(c).is_not_null() for c in cols]))
            y=pl.read_parquet(q).filter((pl.col("decision_time")>=A)&(pl.col("label_end")<B))
            frames.append(x.select("symbol","decision_time","volatility_20").join(y,on=["symbol","decision_time"],validate="1:1"))
        atomic_write(build_paths(pl.concat(frames),raw),labelpath)
    labels=pl.read_parquet(labelpath)
    paths=[]; oldpaths=[]; counts=[]
    for fd in man["folds"]:
        fold=Fold(**fd)
        p=cache/(str(fold.fold_id)+".parquet")
        oldpath=folder/"predictions"/p.name
        old=pl.read_parquet(oldpath)
        if not p.exists():
            train,test=load_fold(pairs,cols,fold)
            train=adapt(train,labels);test=adapt(test,labels)
            assert train["label_end"].max()<fold.test_start
            assert test["label_end"].max()<fold.test_end
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
            f=test.select(*KEYS,"fixed_end","fixed_exit","barrier_width","barrier_hit").with_columns(pl.Series("cost_target",yv),*[pl.Series(n,prob[:,i]) for i,n in enumerate(["p_short","p_flat","p_long"])],*[pl.lit(float(prior[i])).alias(n) for i,n in enumerate(["prior_short","prior_flat","prior_long"])])
            atomic_write(f,p)
        f=pl.read_parquet(p)
        assert f.with_columns(pl.col("fixed_end").alias("label_end"),pl.col("fixed_exit").alias("exit_price")).select(KEYS).equals(old.select(KEYS)),"Common eligibility mismatch"
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
    first=f["decision_time"].min()//DAY*DAY;last=(old["label_end"].max()//DAY+1)*DAY
    for name,t in [("incumbent",pick(old,"logistic",.7)),("triple_barrier",select(f)),("incumbent_with_barrier",pick(adapt(old,labels),"logistic",.7))]:
        rows=[]
        for fee,slip in [(5,1),(10,5)]:
            scored=score_trades(t,fee,slip,fund,{"BTCUSDT":[(A,B)]}) if t.height else t
            rows.append({"fee_bps":fee,"slippage_bps":slip,**metrics(scored,first,last)})
        out["results"].append({"name":name,"economics":rows})
    out["status"]="pass"
def main():
    out={"scope":"Closed-minute trigger symmetric log barrier max(0.0012,2*historical hourly volatility20); next minute open execution; four hour maximum. Net-profit classes at actual exit; conservative original fixed-horizon eligibility/purge retained, actual label end asserted. No intrabar barrier fills, no tuning."}
    try:audit(out)
    except Exception:out.update(status="fail",error=traceback.format_exc())
    p=ROOT/"reports"/"22_triple_barrier.zip";p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(".tmp")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:z.writestr("metrics.json",json.dumps(out,allow_nan=False))
    tmp.replace(p);print(p,out["status"])
if __name__=="__main__":main()
