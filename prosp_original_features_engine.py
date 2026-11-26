"""Bounded ProSP adaptation: pooled logistic tail classification, cached panel."""
import os
os.environ.setdefault("POLARS_MAX_THREADS","1")
import json,zipfile,traceback
from pathlib import Path
import numpy as np
import polars as pl
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits
from baseline_engine import digest,files
from feature_engine import BASE_FEATURES
from validation_protocol import Fold,select_fold
from experiment_store import identity
from data_engine import atomic_write
from robustness_engine import A,B,DAY,ci
ROOT=Path("data");H=3600000
KEYS=["symbol","decision_time","entry_time","label_end","entry_price","exit_price","target_return"]
MIN=60000
WINDOWS=(1,3,6,9,12)
def build(symbol,hours):
    step=hours*H
    raw=sorted(p for p in (ROOT/"futures"/symbol/"1m").glob("*.parquet") if "2021-01"<=p.stem<="2024-12")
    spot=sorted(p for p in (ROOT/"spot"/symbol).glob("*.parquet") if "2021-01"<=p.stem<="2024-12")
    bars=[];points=[];sp=[]
    for p in raw:
        x=pl.read_parquet(p,columns=["open_time","open","close","volume","quote_volume"]).sort("open_time")
        assert np.all(np.diff(x["open_time"].to_numpy())==MIN)
        bars.append(x.with_columns((pl.col("open_time")//step*step+step).alias("decision_time")).group_by("decision_time").agg(pl.len().alias("n"),pl.col("close").last(),pl.col("volume").sum(),pl.col("quote_volume").sum()).filter(pl.col("n")==step//MIN))
        points.append(x.filter(pl.col("open_time")%step==15*MIN).select("open_time","open"))
    f=pl.concat(bars).sort("decision_time")
    assert np.array_equal(f["decision_time"].to_numpy(),np.arange(A+step,B+1,step))
    for p in spot:
        sp.append(pl.scan_parquet(p).filter(pl.col("event_time")%step==step-MIN).select((pl.col("event_time")+MIN).alias("decision_time"),pl.col("close").alias("spot")).collect())
    if sp:f=f.join(pl.concat(sp),on="decision_time",how="left",validate="1:1")
    else:f=f.with_columns(pl.lit(None,dtype=pl.Float64).alias("spot"))
    p=pl.col("close");v=pl.col("quote_volume");s=pl.col("spot")
    f=f.with_columns((p/p.shift(1)-1).alias("ret"),v.alias("turnover"),(p.log()-s.log()).alias("basis"),(p*pl.col("volume")).alias("pv"))
    expr=[]
    for w in WINDOWS:
        expr.extend([(p.shift(1)/p.shift(w+1)-1).alias(f"cr{w}"),v.shift(1).rolling_sum(w).alias(f"co{w}")])
    for w in (3,6,9,12):
        expr.append((p.shift(2)/p.shift(w+1)-1).alias(f"crnorm{w}"))
    for w in range(1,13):expr.append(pl.col("ret").shift(w).alias(f"nr{w}"))
    for day in (7,14,21,28,35):
        k=day*24//hours
        expr.append((p.shift(1)/p.shift(k+1)-s.shift(1)/s.shift(k+1)).alias(f"bm{day}"))
    for w in (3,6,9,12,24):expr.append(pl.col("pv").rolling_var(w,ddof=1).alias(f"pvvar{w}"))
    f=f.with_columns(*expr)
    k=35*24//hours
    valid=(s.is_finite()&(s>0)).fill_null(False).cast(pl.Int64)
    f=f.with_columns((valid.rolling_sum(k+2)==k+2).fill_null(False).alias("spot_ready"))
    pt=pl.concat(points)
    f=f.with_columns(pl.lit(symbol).alias("symbol"),(pl.col("decision_time")+15*MIN).alias("entry_time"),(pl.col("decision_time")+15*MIN+step).alias("label_end"))
    f=f.join(pt.rename({"open_time":"entry_time","open":"entry_price"}),on="entry_time",validate="1:1").join(pt.rename({"open_time":"label_end","open":"exit_price"}),on="label_end",validate="1:1").filter(pl.col("label_end")<B).with_columns((pl.col("exit_price")/pl.col("entry_price")).log().alias("target_return"))
    return f,raw+spot
S1=["ret","turnover"]+[f"{k}{w}" for w in WINDOWS for k in ("cr","co")]+[f"nr{w}" for w in range(1,13)]+[f"crnorm{w}" for w in (3,6,9,12)]
S2=["basis","pv"]+[f"bm{d}" for d in (7,14,21,28,35)]+[f"pvvar{w}" for w in (3,6,9,12,24)]
def run(panel,cols,man,params,seed,cache,out,n):
    assert len(cols) in (28,40)
    norm=[k for k in S1 if k.startswith("nr") or k.startswith("crnorm")]
    panel=panel.with_columns(*[((pl.col(k)-pl.col(k).mean().over("decision_time"))/pl.col(k).std(ddof=0).over("decision_time")).alias(k) for k in norm])
    panel=panel.filter(pl.all_horizontal([pl.col(k).is_finite().fill_null(False) for k in cols])).filter(pl.len().over("decision_time")==n)
    panel=panel.sort(["decision_time","target_return","symbol"]).with_columns(pl.int_range(pl.len()).over("decision_time").alias("rank")).with_columns(pl.when(pl.col("rank")<2).then(-1).when(pl.col("rank")>=n-2).then(1).otherwise(0).alias("target")).sort("decision_time","symbol")
    paths=[]
    for fd in man["folds"]:
        fold=Fold(**fd);train,test=select_fold(panel,fold)
        if train.height<100 or test.height<30:raise ValueError("Insufficient fold coverage")
        p=cache/(str(fold.fold_id)+".parquet")
        if not p.exists():
            scale=StandardScaler()
            with threadpool_limits(limits=1):
                x=scale.fit_transform(train.select(cols).to_numpy());v=scale.transform(test.select(cols).to_numpy())
                model=LogisticRegression(C=params["C"],max_iter=500,random_state=seed).fit(x,train["target"].to_numpy())
                assert np.array_equal(model.classes_,[-1,0,1])
                prob=model.predict_proba(v)
            atomic_write(test.select(*KEYS,"target").with_columns(*[pl.Series(k,prob[:,i]) for i,k in enumerate(["p_bottom","p_middle","p_top"])]),p)
        f=pl.read_parquet(p);assert f.select(*KEYS,"target").equals(test.select(*KEYS,"target"))
        paths.append(p)
    f=pl.read_parquet(paths).sort("decision_time","symbol");q=f
    prob=f.select("p_bottom","p_middle","p_top").to_numpy();y=f["target"].to_numpy()[:,None]==np.array([-1,0,1])
    out.update({"balanced_periods":f.height//n,"brier":float(np.mean(np.sum((prob-y)**2,axis=1))),"naive_brier":float(np.mean(np.sum((np.array([2/n,1-4/n,2/n])-y)**2,axis=1))),"results":[]})
    for name in ("dual_sort",):
        records=[];overlap=0;flat=0
        for (d,),g in q.group_by("decision_time",maintain_order=True):
            if name=="dual_sort":
                l=set(g.sort(["p_top","symbol"],descending=[True,False]).head(2)["symbol"].to_list())
                s=set(g.sort(["p_bottom","symbol"],descending=[True,False]).head(2)["symbol"].to_list())
                both=l&s;overlap+=len(both);l-=both;s-=both
            else:
                order=g.with_columns((pl.col("p_top")-pl.col("p_bottom")).alias("score")).sort(["score","symbol"])
                l=set(order.tail(2)["symbol"].to_list());s=set(order.head(2)["symbol"].to_list())
            if not l:flat+=1
            assert len(l)==len(s)
            # Preserve calendar and cash on empty baskets; no replacement selections.
            z={"decision_time":int(d),"label_end":int(g["label_end"][0]),"legs":len(l)+len(s)}
            for fee,slip in [(0,0),(1,0),(5,1),(10,5)]:
                total=0.
                for row in g.iter_rows(named=True):
                    side=1 if row["symbol"] in l else -1 if row["symbol"] in s else 0
                    if not side:continue
                    a=row["entry_price"]*(1+side*slip/10000);b=row["exit_price"]*(1-side*slip/10000)
                    total+=(side*(b-a)/a-fee/10000*(a+b)/a)/(2*len(l))
                z[f"r_{fee}_{slip}"]=total
            records.append(z)
        basket=pl.DataFrame(records).sort("decision_time")
        atomic_write(basket,cache/(name+".parquet"))
        rows=[];first=int(basket["decision_time"].min())//DAY*DAY;last=(int(basket["label_end"].max())//DAY+1)*DAY
        ix=(basket["label_end"].to_numpy()-first)//DAY
        counts=np.bincount(ix,minlength=(last-first)//DAY)
        for fee,slip in [(0,0),(1,0),(5,1),(10,5)]:
            ret=basket[f"r_{fee}_{slip}"].to_numpy();eq=np.r_[0,np.cumsum(ret)];loss=-ret[ret<0].sum()
            daily=np.bincount(ix,weights=ret,minlength=len(counts))
            rows.append({"fee_bps":fee,"slippage_bps":slip,"expectancy":float(ret.mean()),"pnl":float(ret.sum()),"profit_factor":float(ret[ret>0].sum()/loss) if loss else None,"fixed_notional_drawdown":float(max(np.maximum.accumulate(eq)-eq)),"bootstrap":[ci(daily,counts,b) for b in (7,30)]})
        out["results"].append({"name":name,"baskets":len(records),"flat_baskets":flat,"removed_overlap_pairs":overlap,"mean_legs":float(basket["legs"].mean()),"fee_slip_only":rows})

def audit(out):
    with zipfile.ZipFile(ROOT/"reports"/"29_prosp_readiness.zip") as z:r=json.loads(z.read("metrics.json"))
    assert r["status"]=="pass"
    folder,=list((ROOT/"experiments").glob("6823506d*"))
    spec=json.loads((folder/"spec.json").read_text());man=json.loads((folder/"manifest.json").read_text())
    params=json.loads((folder/"result.json").read_text())["params"]
    out["runs"]=[]
    for hours in (4,8):
        frames=[];sources=[]
        for row in r["results"]:
            f,src=build(row["symbol"],hours);frames.append(f);sources+=src
        panel=pl.concat(frames).sort("decision_time","symbol")
        common=panel.filter((pl.col("symbol")!="ATOMUSDT")&pl.col("spot_ready")).filter(pl.len().over("decision_time")==9)
        common=common.filter(pl.all_horizontal([pl.col(k).is_finite().fill_null(False) for k in S1+S2])).filter(pl.len().over("decision_time")==9)
        source=digest(sources);code=digest([Path(__file__),Path("validation_protocol.py")])
        for name,p,cols,n in [("set1_10",panel,S1,10),("set1_9",common,S1,9),("set3_9",common,S1+S2,9)]:
            row={"hours":hours,"name":name};out["runs"].append(row)
            key=identity({"source":source,"code":code,"hours":hours,"name":name,"seed":spec["seed"],"params":params,"sklearn":sklearn.__version__,"polars":pl.__version__})
            row["experiment"]=key
            try:run(p,cols,man,params,spec["seed"],ROOT/"prosp_original"/key,row,n);row["status"]="pass"
            except Exception:row.update(status="fail",error=traceback.format_exc())
            print(hours,name,row["status"],flush=True)
    out["status"]="pass" if all(r["status"]=="pass" for r in out["runs"]) else "partial"
def main():
    out={"scope":"Paper-inspired 28/40 features; quote volume turnover proxy, ordinary sample variance for equation 6, fixed 4h/8h bars, nine-asset common spot sample and ten-asset Set1 control; fee/slippage only; no funding, tuning or holdout"}
    try:audit(out)
    except Exception:out.update(status="fail",error=traceback.format_exc())
    p=ROOT/"reports"/"30_prosp_features.zip";p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(".tmp")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:z.writestr("metrics.json",json.dumps(out,allow_nan=False))
    tmp.replace(p);print(p,out["status"])
if __name__=="__main__":main()
