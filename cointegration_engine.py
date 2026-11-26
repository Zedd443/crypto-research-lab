"""Train-only Engle-Granger screen, BTC reference, development only."""
import os
os.environ.setdefault("POLARS_MAX_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import json,zipfile,traceback
from pathlib import Path
import numpy as np
import polars as pl
from statsmodels.tsa.stattools import coint,adfuller
from statsmodels.stats.multitest import multipletests
import statsmodels
from threadpoolctl import threadpool_limits
from baseline_engine import digest
from robustness_engine import A,B
from experiment_store import identity
from data_engine import write_json
ROOT=Path("data");H=3600000
SYMBOLS=["BTCUSDT","ETHUSDT","BNBUSDT","DOGEUSDT","ADAUSDT","LINKUSDT","AVAXUSDT","BCHUSDT","UNIUSDT","ATOMUSDT"]
def load(symbol):
    paths=sorted(p for p in (ROOT/"futures"/symbol/"1m").glob("*.parquet") if "2021-01"<=p.stem<="2024-12")
    assert len(paths)==48
    frames=[]
    for p in paths:
        f=pl.read_parquet(p,columns=["open_time","close"]).sort("open_time")
        ts=f["open_time"].to_numpy()
        assert len(ts)>0 and np.all(np.diff(ts)==60000) and np.all(ts%60000==0)
        frames.append(f.filter(pl.col("open_time")%H==H-60000).select((pl.col("open_time")+60000).alias("decision_time"),"close"))
    f=pl.concat(frames).sort("decision_time")
    assert np.array_equal(f["decision_time"].to_numpy(),np.arange(A+H,B+1,H))
    v=f["close"].to_numpy()
    assert np.isfinite(v).all() and (v>0).all()
    return f["decision_time"].to_numpy(),v,paths
def screen(y,x):
    # Price-level spread with intercept; no log-return substitution.
    design=np.column_stack([np.ones(len(x)),x])
    alpha,beta=np.linalg.lstsq(design,y,rcond=None)[0]
    spread=y-alpha-beta*x
    stat,p,_=coint(y,x,trend="c",maxlag=24,autolag="AIC")
    lag=spread[:-1];delta=np.diff(spread)
    slope=np.linalg.lstsq(np.column_stack([np.ones(len(lag)),lag]),delta,rcond=None)[0][1]
    phi=1+slope
    half=float(-np.log(2)/np.log(phi)) if 0<phi<1 else None
    def adf(z):return float(adfuller(z,maxlag=24,autolag="AIC",regression="c")[1])
    return {"alpha":float(alpha),"beta":float(beta),"coint_stat":float(stat),"p":float(p),"spread_std":float(spread.std(ddof=1)),"phi":float(phi),"half_life_hours":half,"level_adf_p":[adf(y),adf(x)],"difference_adf_p":[adf(np.diff(y)),adf(np.diff(x))]}
def audit(out):
    folder,=list((ROOT/"experiments").glob("6823506d*"))
    man=json.loads((folder/"manifest.json").read_text())
    t,btc,src=load(SYMBOLS[0]);prices={};sources=list(src)
    for s in SYMBOLS[1:]:
        ts,v,paths=load(s);assert np.array_equal(ts,t);prices[s]=v;sources+=paths
    source=digest(sources);code=digest([Path(__file__)])
    key=identity({"source":source,"code":code,"manifest":man["manifest_id"],"statsmodels":statsmodels.__version__})
    cache=ROOT/"pairs_screen"/key
    rows=[]
    for fd in man["folds"]:
        mask=(t>=fd["train_start"])&(t<fd["test_start"])
        y=btc[mask]
        p=cache/(str(fd["fold_id"])+".json")
        if p.exists():group=json.loads(p.read_text())
        else:
            group=[]
            with threadpool_limits(limits=1):
                for s,x in prices.items():
                    r=screen(y,x[mask]);r.update(symbol=s,fold=fd["fold_id"],train_rows=int(mask.sum()))
                    group.append(r)
            adjusted=multipletests([r["p"] for r in group],alpha=.05,method="holm")[1]
            for r,padj in zip(group,adjusted):
                r["holm_p"]=float(padj)
                r["i1_screen"]=all(p>.05 for p in r["level_adf_p"]) and all(p<.05 for p in r["difference_adf_p"])
                r["eligible"]=bool(padj<.05 and r["i1_screen"] and r["beta"]>0 and r["spread_std"]>0 and r["half_life_hours"] is not None)
            write_json(p,group)
        rows+=group;print("fold",fd["fold_id"],"done",flush=True)
    out.update(experiment=key,source=source,code=code,folds=len(man["folds"]),rows=rows,summary=[])
    for s in SYMBOLS[1:]:
        g=[r for r in rows if r["symbol"]==s]
        h=[r["half_life_hours"] for r in g if r["eligible"]]
        out["summary"].append({"symbol":s,"eligible_folds":len(h),"raw_coint_pass":sum(r["p"]<.05 for r in g),"holm_pass":sum(r["holm_p"]<.05 for r in g),"i1_pass":sum(r["i1_screen"] for r in g),"median_half_life_hours":float(np.median(h)) if h else None})
    out["status"]="pass"
def main():
    out={"scope":"BTC-reference price-level Engle-Granger pilot, 1h completed futures closes, existing 180d training windows, no test-period fitting, within-fold Holm across nine pairs, ADF I(1) diagnostic, half-life AR1 approximation; no backtest or profitability claim; no holdout"}
    try:audit(out)
    except Exception:out.update(status="fail",error=traceback.format_exc())
    p=ROOT/"reports"/"32_cointegration.zip";p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(".tmp")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:z.writestr("metrics.json",json.dumps(out,allow_nan=False))
    tmp.replace(p);print(p,out["status"])
if __name__=="__main__":main()
