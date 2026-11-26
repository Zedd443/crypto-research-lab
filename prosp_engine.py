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
def audit(out):
    with zipfile.ZipFile(ROOT/"reports"/"25_multi_asset_transfer.zip") as z:r=json.loads(z.read("metrics.json"))
    assert r["status"]=="pass" and len(r["results"])==10
    folder,=list((ROOT/"experiments").glob("6823506d*"))
    spec=json.loads((folder/"spec.json").read_text());man=json.loads((folder/"manifest.json").read_text())
    cols=spec["features"];assert set(cols)==set(BASE_FEATURES)-{"body"}
    params=json.loads((folder/"result.json").read_text())["params"]
    frames=[];sources=[]
    for row in r["results"]:
        s=row["symbol"]
        if s=="BTCUSDT":
            pairs=[(p,q) for p,q in files(ROOT,[s],"1h",4,15) if "2021-01"<=p.stem<="2024-12"]
            assert len(pairs)==48
            f=pl.concat([pl.read_parquet(p,columns=["symbol","decision_time",*BASE_FEATURES]).join(pl.read_parquet(q).select(KEYS),on=["symbol","decision_time"],validate="1:1") for p,q in pairs])
            sources.extend(v for pair in pairs for v in pair)
        else:
            p=ROOT/"multi_asset"/row["experiment"]/"dataset.parquet"
            f=pl.read_parquet(p);sources.append(p)
        f=f.filter((pl.col("decision_time")>=A)&(pl.col("label_end")<B)&pl.all_horizontal([pl.col(k).is_finite().fill_null(False) for k in BASE_FEATURES+["target_return"]]))
        assert f["symbol"].unique().to_list()==[s] and f["decision_time"].n_unique()==f.height
        assert ((f["entry_time"]-f["decision_time"])==15*60000).all() and ((f["label_end"]-f["entry_time"])==4*H).all()
        frames.append(f.select(*KEYS,*cols))
    panel=pl.concat(frames).filter(pl.len().over("decision_time")==10).sort("decision_time","symbol")
    # Two extreme assets per tail (floor of 25% of ten); deterministic tie handling.
    panel=panel.sort(["decision_time","target_return","symbol"]).with_columns(pl.int_range(pl.len()).over("decision_time").alias("rank")).with_columns(pl.when(pl.col("rank")<2).then(-1).when(pl.col("rank")>=8).then(1).otherwise(0).alias("target")).drop("rank").sort("decision_time","symbol")
    source=digest(sources);code=digest([Path(__file__),Path("validation_protocol.py")])
    key=identity({"source":source,"code":code,"incumbent":folder.name,"params":params,"seed":spec["seed"],"sklearn":sklearn.__version__,"polars":pl.__version__})
    cache=ROOT/"prosp"/key;paths=[]
    for fd in man["folds"]:
        fold=Fold(**fd);train,test=select_fold(panel,fold)
        p=cache/(str(fold.fold_id)+".parquet")
        if not p.exists():
            imp=SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True)
            x=imp.fit_transform(train.select(cols).to_numpy());v=imp.transform(test.select(cols).to_numpy())
            scale=StandardScaler()
            with threadpool_limits(limits=1):
                x=scale.fit_transform(x);v=scale.transform(v)
                model=LogisticRegression(C=params["C"],max_iter=500,random_state=spec["seed"]).fit(x,train["target"].to_numpy())
                assert np.array_equal(model.classes_,[-1,0,1])
                prob=model.predict_proba(v)
            atomic_write(test.select(*KEYS,"target").with_columns(*[pl.Series(n,prob[:,i]) for i,n in enumerate(["p_bottom","p_middle","p_top"])]),p)
        f=pl.read_parquet(p)
        assert f.select(*KEYS,"target").equals(test.select(*KEYS,"target"))
        paths.append(p);print("fold",fold.fold_id,flush=True)
    f=pl.read_parquet(paths).sort("decision_time","symbol")
    assert f.select("symbol","decision_time").unique().height==f.height and f["label_end"].max()<B
    assert (f.group_by("decision_time").len()["len"]==10).all()
    prob=f.select("p_bottom","p_middle","p_top").to_numpy()
    assert np.isfinite(prob).all() and np.allclose(prob.sum(1),1)
    y=(f["target"].to_numpy()[:,None]==np.array([-1,0,1]))
    q=f.filter(pl.col("decision_time")%(4*H)==0)
    out.update({"experiment":key,"source":source,"code":code,"balanced_hours":f.height//10,"scheduled_hours":q.height//10,"brier":float(np.mean(np.sum((prob-y)**2,axis=1))),"naive_brier":float(np.mean(np.sum((np.array([.2,.6,.2])-y)**2,axis=1))),"results":[]})
    for name in ("dual_sort","single_score"):
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
    out["status"]="pass"
def main():
    out={"scope":"ProSP-inspired pooled multinomial logistic; fixed 10 survivor assets; two realized tail labels per side, dual sort remove overlaps, equal weight surviving legs, cash if none; single-score comparator; fixed 4h UTC schedule, full roundtrip costs, no funding or holdout; no tuning; not original paper replication"}
    try:audit(out)
    except Exception:out.update(status="fail",error=traceback.format_exc())
    p=ROOT/"reports"/"28_prosp.zip";p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(".tmp")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:z.writestr("metrics.json",json.dumps(out,allow_nan=False))
    tmp.replace(p);print(p,out["status"])
if __name__=="__main__":main()
