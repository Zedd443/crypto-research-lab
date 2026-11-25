"""Development-only predictive baseline. No final holdout access or economic claims."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault("POLARS_MAX_THREADS","1")
import polars as pl
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, brier_score_loss, balanced_accuracy_score, confusion_matrix, log_loss
from threadpoolctl import threadpool_limits
import sklearn
import data_engine as de
from feature_engine import BASE_FEATURES, FEATURES, STEPS, HORIZONS_H
from validation_protocol import make_folds, manifest, select_fold
from experiment_store import checkpoint, identity, tune, report_fold
from resource_policy import budget
from universe import SYMBOLS

def files(root,symbols,tf,h):
    pairs=[]
    for s in symbols:
        for p in sorted((root/"features"/s/tf).glob("*.parquet")):
            q=root/"labels"/s/tf/str(h)/p.name
            if q.exists():pairs.append((p,q))
    if not pairs:raise ValueError("Build features and labels first")
    return pairs

def digest(paths):
    h=hashlib.sha256()
    for p in sorted(set(paths),key=str):
        h.update(str(p).encode())
        with p.open("rb") as f:
            for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def load_fold(pairs,cols,fold):
    frames=[]
    for p,q in pairs:
        # Predicate pushdown; no holdout rows enter any development fold.
        cond=pl.col("decision_time").is_between(fold.train_start,fold.test_end,closed="left")
        x=pl.scan_parquet(p).filter(cond).select("symbol","decision_time",*cols)
        y=pl.scan_parquet(q).filter(cond & (pl.col("label_end")<fold.test_end)).select("symbol","decision_time","label_end","target_return")
        f=x.join(y,on=["symbol","decision_time"],how="inner",validate="1:1").collect()
        if f.height:frames.append(f)
    if not frames:raise ValueError("Empty fold")
    f=pl.concat(frames).filter(pl.col("target_return").is_finite())
    # Warm-up required for the core features, optional auxiliaries may be null.
    f=f.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in BASE_FEATURES]))
    train,test=select_fold(f,fold)
    if train.height<100 or test.height<30:raise ValueError("Insufficient valid fold rows")
    return train,test

def fit_score(train,test,cols,model,params,threads,seed):
    # XGBoost early stopping uses only an internal, purged chronological slice.
    fit=train
    inner=None
    if model=="xgb":
        cut=int(train["decision_time"].quantile(.85))
        fit=train.filter((pl.col("decision_time")<cut)&(pl.col("label_end")<cut))
        inner=train.filter(pl.col("decision_time")>=cut)
        if fit.height<100 or inner.height<30:raise ValueError("Insufficient inner validation rows")
    X=fit.select(cols).to_numpy(); Xt=test.select(cols).to_numpy()
    imp=SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True)
    X=imp.fit_transform(X); Xt=imp.transform(Xt)
    y=fit["target_return"].to_numpy(); yt=test["target_return"].to_numpy()
    with threadpool_limits(limits=threads):
        if model in ("ridge","logistic"):
            scaler=StandardScaler()
            X=scaler.fit_transform(X); Xt=scaler.transform(Xt)
        if model=="ridge":
            estimator=Ridge(alpha=params["alpha"]).fit(X,y)
        elif model=="logistic":
            binary=(y>0).astype(int)
            if len(np.unique(binary))<2:raise ValueError("Single-class training fold")
            estimator=LogisticRegression(C=params["C"],max_iter=500,random_state=seed).fit(X,binary)
            pred=estimator.predict_proba(Xt)[:,1]
            target=(yt>0).astype(int)
            base=np.full(len(yt),binary.mean())
            loss=brier_score_loss(target,pred)
            return {"score":-float(loss),"loss":float(loss),"baseline_loss":float(brier_score_loss(target,base)),
                    "direction_accuracy":float(((pred>=.5)==target).mean()),
                    "balanced_accuracy":float(balanced_accuracy_score(target,pred>=.5)),
                    "confusion_matrix":confusion_matrix(target,pred>=.5,labels=[0,1]).tolist(),
                    "log_loss":float(log_loss(target,pred,labels=[0,1])),
                    "baseline_direction_accuracy":float(((base>=.5)==target).mean()),"train_rows":len(y),"test_rows":len(yt)}
        elif model=="rf":
            estimator=RandomForestRegressor(**params,n_jobs=threads,random_state=seed).fit(X,y)
        elif model=="xgb":
            from xgboost import XGBRegressor
            Xi=imp.transform(inner.select(cols).to_numpy())
            yi=inner["target_return"].to_numpy()
            estimator=XGBRegressor(**params,n_jobs=threads,random_state=seed,tree_method="hist",
                                    early_stopping_rounds=20,eval_metric="rmse")
            estimator.fit(X,y,eval_set=[(Xi,yi)],verbose=False)
        else:raise ValueError("Unknown model")
        pred=estimator.predict(Xt)
    loss=mean_squared_error(yt,pred)
    return {"score":-float(loss),"loss":float(loss),"baseline_loss":float(mean_squared_error(yt,np.zeros_like(yt))),
            "mae":float(mean_absolute_error(yt,pred)),
            "rmse":float(np.sqrt(loss)),
            "oos_r2_zero":float(1-loss/np.mean(yt**2)) if np.mean(yt**2)>0 else None,
            "direction_accuracy":float(((pred>0)==(yt>0)).mean()),
            "train_rows":len(y),"test_rows":len(yt)}

def parameters(model,trial=None):
    if model=="ridge":return {"alpha":trial.suggest_float("alpha",.01,100,log=True) if trial else 1.}
    if model=="logistic":return {"C":trial.suggest_float("C",.01,10,log=True) if trial else 1.}
    if model=="rf":return {"n_estimators":trial.suggest_categorical("n_estimators",[50,100,200]) if trial else 100,
        "max_depth":trial.suggest_int("max_depth",3,12) if trial else 6,
        "min_samples_leaf":trial.suggest_int("min_samples_leaf",5,40) if trial else 20}
    return {"n_estimators":300,"max_depth":trial.suggest_int("max_depth",2,6) if trial else 3,
            "learning_rate":trial.suggest_float("learning_rate",.01,.2,log=True) if trial else .05,
            "subsample":.8,"colsample_bytree":.8}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path("data"))
    p.add_argument("--symbols",nargs="+",choices=SYMBOLS,default=["BTCUSDT"])
    p.add_argument("--tf",choices=list(STEPS),default="1h")
    p.add_argument("--horizon",type=int,choices=HORIZONS_H,default=4)
    p.add_argument("--start",required=True)
    p.add_argument("--holdout-start",required=True)
    p.add_argument("--holdout-end",required=True)
    p.add_argument("--train-days",type=int,default=180)
    p.add_argument("--test-days",type=int,default=30)
    p.add_argument("--model",choices=["ridge","logistic","rf","xgb"],default="ridge")
    p.add_argument("--aux",action="store_true")
    p.add_argument("--trials",type=int,default=0)
    p.add_argument("--threads",type=int,default=2,choices=[1,2,4])
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    if a.trials<0:p.error("Negative trial count")
    if a.horizon*3600000%STEPS[a.tf]:p.error("Horizon must be a multiple of timeframe")
    start,hs,he=map(lambda v:de.ms(de.date(v)),(a.start,a.holdout_start,a.holdout_end))
    folds=make_folds(start,hs,a.train_days,a.test_days)
    if not folds:p.error("No development folds")
    pairs=files(a.root,list(dict.fromkeys(a.symbols)),a.tf,a.horizon)
    # Exclude partitions strictly after the development period from identity as well.
    pairs=[(x,y) for x,y in pairs if de.ms(de.date(x.stem+"-01"))<hs]
    if not pairs:p.error("No development data")
    source=digest([f for pair in pairs for f in pair])
    protocol=manifest(folds,hs,he,source)
    code=digest([Path(__file__),Path(__file__).with_name("feature_engine.py"),Path(__file__).with_name("validation_protocol.py")])
    cols=FEATURES if a.aux else BASE_FEATURES
    # One coordinator, bounded internal threads. No concurrent copies of fold matrices.
    budget(1,per_worker_bytes=2*1024**3,threads_per_worker=a.threads)
    spec={"manifest_id":protocol["manifest_id"],"code":code,"sklearn":sklearn.__version__,
          "polars":pl.__version__,"model":a.model,"features":cols,"tf":a.tf,"horizon":a.horizon,
          "seed":a.seed,"threads":a.threads,"objective":"development_brier" if a.model=="logistic" else "development_mse"}
    if a.model=="xgb":
        import xgboost
        spec["xgboost"]=xgboost.__version__
    out=a.root/"experiments"/identity(spec)
    de.write_json(out/"manifest.json",protocol)
    de.write_json(out/"spec.json",spec)
    def evaluate(params,trial=None):
        scores=[]
        for fold in folds:
            fspec={**spec,"params":params}
            def compute():
                train,test=load_fold(pairs,cols,fold)
                return fit_score(train,test,cols,a.model,params,a.threads,a.seed)
            result=checkpoint(out/"folds",fspec,fold.fold_id,compute)
            scores.append(result["score"])
            print(a.model,"fold",fold.fold_id,"score",result["score"],flush=True)
            if trial is not None:report_fold(trial,fold.fold_id,float(np.mean(scores)))
        return float(np.mean(scores))
    if a.trials:
        study=tune(out/"studies",spec,lambda trial:evaluate(parameters(a.model,trial),trial),a.trials,a.seed)
        result={"best_score":study.best_value,"best_params":study.best_params,"trial_count":len(study.trials)}
    else:
        params=parameters(a.model)
        result={"score":evaluate(params),"params":params}
    result["note"]="Development diagnostic only; no final holdout or economic execution evaluated. Tuning scores are selection scores, not unbiased OOS."
    de.write_json(out/"result.json",result)
    print(json.dumps(result),flush=True)
if __name__=="__main__":main()
