"""Development-only BTC robustness screening."""
import json,zipfile
from pathlib import Path
import numpy as np
import polars as pl
from data_engine import ms,date
from execution_engine import score_trades
from experiment_store import identity

DAY=86400000
A=ms(date("2021-01-01"))
B=ms(date("2025-01-01"))

def ledger(root):
    parts=[]
    for p in sorted((root/"funding"/"BTCUSDT").glob("*.parquet")):
        if not "2021-01"<=p.stem<="2024-12":continue
        f=pl.read_parquet(p)
        m=pl.read_parquet(root/"mark"/"BTCUSDT"/p.name,columns=["event_time","open"]).rename({"event_time":"minute","open":"mark_price"})
        parts.append(f.with_columns((pl.col("event_time")//60000*60000).alias("minute")).join(m,on="minute",how="left",validate="m:1"))
    f=pl.concat(parts).filter(pl.col("event_time").is_between(A,B,closed="left")).sort("event_time")
    expected=np.arange(A,B,8*3600000)
    t=f["event_time"].to_numpy()
    grid=((t+30000)//(8*3600000))*(8*3600000)
    diagnostic={"actual_events":len(t),"expected_events":len(expected),"missing_grid":np.setdiff1d(expected,grid).tolist(),"unexpected_grid":np.setdiff1d(grid,expected).tolist(),"duplicate_grid":len(grid)-len(np.unique(grid)),"off_grid_events":t[np.abs(t-grid)>60000].tolist(),"missing_mark_times":f.filter(pl.col("mark_price").is_null())["event_time"].to_list(),"null_interval_times":f.filter(pl.col("funding_interval_hours").is_null())["event_time"].to_list(),"non_8h_intervals":f.filter(pl.col("funding_interval_hours")!=8).select("event_time","funding_interval_hours").to_dicts()}
    from data_engine import write_json
    write_json(root/"reports"/"funding_diagnostic.json",diagnostic)
    if len(t)!=len(expected) or np.any(abs(t-expected)>60000) or f["funding_interval_hours"].null_count() or not (f["funding_interval_hours"]==8).all():
        raise ValueError("Funding schedule validation failed")
    return f.with_columns(pl.lit("BTCUSDT").alias("symbol")).select("symbol","event_time","funding_rate","mark_price")

def pick(f,model,th):
    p=f["prediction"].to_numpy()
    s=np.where(p>=th,1,np.where(p<=1-th,-1,0)) if model=="logistic" else np.where(p>th,1,np.where(p< -th,-1,0))
    ix=[];free=-1
    for i,(d,e,v) in enumerate(zip(f["decision_time"],f["label_end"],s)):
        if v and d>=free:ix.append(i);free=e
    return f.with_columns(pl.Series("side",s)).with_row_index("idx").filter(pl.col("idx").is_in(ix)).drop("idx")

def ci(p,c,b):
    rng=np.random.default_rng(42);n=len(p);v=[]
    for _ in range(10000):
        ix=((rng.integers(0,n,size=(n+b-1)//b)[:,None]+np.arange(b))%n).ravel()[:n]
        k=c[ix].sum()
        if k:v.append(p[ix].sum()/k)
    return {"block_days":b,"valid_reps":len(v),"interval":np.quantile(v,[.025,.975]).tolist() if v else None}

def metrics(t,first,last):
    if not t.height:return {"trades":0}
    r=t["net_return"].to_numpy();n=(last-first)//DAY
    ix=(t["label_end"].to_numpy()-first)//DAY
    p=np.bincount(ix,weights=r,minlength=n);c=np.bincount(ix,minlength=n)
    eq=np.r_[0,np.cumsum(r)];loss=-r[r<0].sum()
    return {"trades":len(r),"expectancy":float(r.mean()),"pnl":float(r.sum()),"win_rate":float((r>0).mean()),"profit_factor":float(r[r>0].sum()/loss) if loss else None,"fixed_notional_drawdown":float(max(np.maximum.accumulate(eq)-eq)),"funding":float(t["funding_return"].sum()),"annual":t.with_columns(pl.from_epoch("label_end",time_unit="ms").dt.year().alias("year")).group_by("year").agg(pl.len().alias("trades"),pl.col("net_return").sum().alias("pnl")).sort("year").to_dicts(),"bootstrap":[ci(p,c,b) for b in (7,30)]}

def main():
    root=Path("data");fund=ledger(root);results=[]
    missing=fund.filter(pl.col("mark_price").is_null())["event_time"].to_numpy()
    fund=fund.filter(pl.col("mark_price").is_not_null())
    for p in sorted((root/"experiments").glob("*/result.json")):
        folder=p.parent;s=json.loads((folder/"spec.json").read_text())
        if not s.get("prediction_export") or s["model"] not in ("logistic","xgb"):continue
        paths=list((folder/"predictions").glob("*.parquet"))
        key=identity({**s,"params":json.loads(p.read_text())["params"]})
        checks=list((folder/"folds"/key).glob("*.json"))
        if not checks or {x.stem for x in checks}!={x.stem for x in paths}:raise ValueError("Incomplete exports")
        for c in checks:
            if pl.read_parquet(folder/"predictions"/(c.stem+".parquet")).height!=json.loads(c.read_text())["result"]["test_rows"]:raise ValueError("Row mismatch")
        f=pl.read_parquet(paths).sort("decision_time")
        if f["symbol"].unique().to_list()!=["BTCUSDT"] or f["decision_time"].n_unique()!=f.height or f["decision_time"].min()<A or f["label_end"].max()>=B:raise ValueError("Invalid scope")
        first=f["decision_time"].min()//DAY*DAY;last=(f["label_end"].max()//DAY+1)*DAY
        rows=[]
        for th in ([.5,.52,.55,.6,.65,.7] if s["model"]=="logistic" else [0,.0005,.001,.002,.005]):
            t=pick(f,s["model"],th)
            total=t.height
            entry=t["entry_time"].to_numpy();end=t["label_end"].to_numpy()
            affected=np.searchsorted(missing,end,side="right")>np.searchsorted(missing,entry,side="right")
            t=t.filter(pl.Series("eligible",~affected))
            for fee,slip in [(2,0),(5,1),(5,3),(10,5)]:
                scored=score_trades(t,fee,slip,fund,{"BTCUSDT":[(A,B)]}) if t.height else t
                rows.append({"threshold":th,"fee_bps":fee,"slippage_bps":slip,"original_trades":total,"excluded_unknown_funding":int(affected.sum()),**metrics(scored,first,last)})
        results.append({"id":folder.name,"spec":s,"rows":rows});print(s["model"],s["horizon"],"done",flush=True)
    if not results:raise ValueError("No experiments")
    out=root/"reports"/"05_robustness.zip";out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix(".tmp")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:z.writestr("metrics.json",json.dumps({"funding_events":fund.height+len(missing),"missing_mark_events":len(missing),"note":"Exploratory only; minute-open mark proxy; trades crossing missing mark events excluded after original signal selection (no replacement trades); results conditional on observed funding marks, funding (entry,exit], assumed costs, marginal bootstrap intervals not adjusted for multiple testing, fixed notional not portfolio equity.","experiments":results},allow_nan=False))
    tmp.replace(out);print(out)
if __name__=="__main__":main()
