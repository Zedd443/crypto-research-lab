"""Read-only validation bundles; never include bulk raw market data."""
import argparse
import json
import math
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import polars as pl
from universe import SYMBOLS

def stats(frame, col):
    s=frame[col].cast(pl.Float64)
    v=s.filter(s.is_finite()).drop_nulls()
    def safe(x):
        return float(x) if x is not None and math.isfinite(x) else None
    return {"rows":len(s),"nulls":s.null_count(),"nonfinite":len(s)-s.null_count()-len(v),
            "mean":safe(v.mean()),"std":safe(v.std()),
            "quantiles":{str(q):safe(v.quantile(q)) for q in (.01,.05,.5,.95,.99)},
            "skew":safe(v.skew()),"kurtosis_excess":safe(v.kurtosis())} if len(v) else {"rows":len(s),"nulls":s.null_count(),"nonfinite":len(s)-s.null_count(),"mean":None}

def data_report(root):
    result={}
    for symbol in SYMBOLS:
        folder=root/"metadata"/symbol
        months=[]
        for p in sorted(folder.glob("????-??.json")):
            m=json.loads(p.read_text())
            if "requested" in m:
                a,b=m["requested"]
                gaps=m.get("missing_ranges_ms",[])
                months.append({"month":p.stem,"rows":m["rows"],"requested_minutes":(b-a)//60000,
                               "missing_minutes":sum((y-x)//60000 for x,y in gaps),
                               "gap_count":len(gaps),"max_gap_minutes":max(((y-x)//60000 for x,y in gaps),default=0)})
        result[symbol]={"months":months,"rows":sum(m["rows"] for m in months),
                        "status":"WARNING" if not months or any(m["missing_minutes"] for m in months) else "PASS",
                        "note":"Requested coverage only; listing and session calendar not verified"}
    paths=sorted((root/"resampled"/"BTCUSDT"/"1h").glob("*.parquet"))
    eda={}
    if paths:
        f=pl.read_parquet(paths).sort("open_time")
        f=f.with_columns(pl.when(pl.col("open_time").diff()==3600000).then(pl.col("close").log().diff()).alias("return"))
        eda=stats(f,"return")
        eda["acf"]={str(k):f.select(pl.corr("return",pl.when(pl.col("open_time")-pl.col("open_time").shift(k)==k*3600000).then(pl.col("return").shift(k)))).item() for k in (1,4,24)}
        eda["acf"]={k:v if v is not None and math.isfinite(v) else None for k,v in eda["acf"].items()}
    return {"symbols":result,"btc_1h_eda":eda,"status":"WARNING","note":"Metadata coverage is not independent raw verification; calendar audit pending."}

def feature_report(root,start,end):
    from feature_engine import FEATURES
    import data_engine as de
    a,b=de.ms(de.date(start)),de.ms(de.date(end))
    paths=sorted((root/"features"/"BTCUSDT"/"1h").glob("*.parquet"))
    if not paths:raise ValueError("Build BTC features first")
    f=pl.scan_parquet(paths).filter(pl.col("decision_time").is_between(a,b,closed="left")).collect().sort("decision_time")
    if not f.height:raise ValueError("No development features")
    cols=[c for c in FEATURES if c in f.columns]
    pairs=[]
    for i,c in enumerate(cols):
        for d in cols[i+1:]:
            x=f.select(pl.corr(c,d)).item()
            if x is not None and math.isfinite(x) and abs(x)>=.95:pairs.append({"a":c,"b":d,"pearson":x})
    labels={}
    for folder in sorted((root/"labels"/"BTCUSDT"/"1h").glob("*")):
        paths=sorted(folder.glob("*.parquet"))
        if not paths:continue
        y=pl.scan_parquet(paths).filter((pl.col("decision_time")>=a)&(pl.col("label_end")<b)).collect()
        if not y.height:continue
        h=int(folder.name)
        invalid=y.filter((pl.col("entry_time")-pl.col("decision_time")!=3600000)|(pl.col("label_end")-pl.col("entry_time")!=h*3600000)|(~pl.col("target_return").is_finite())|((pl.col("target_return")-(pl.col("exit_price")/pl.col("entry_price")).log()).abs()>1e-10)).height
        labels[folder.name]={"stats":stats(y,"target_return"),"positive":y.filter(pl.col("target_return")>0).height,
            "negative":y.filter(pl.col("target_return")<0).height,"zero":y.filter(pl.col("target_return")==0).height,
            "invalid":invalid,"duplicates":y.height-y["decision_time"].n_unique()}
    duplicates=f.height-f["decision_time"].n_unique()
    ages={c:{"negative":f.filter(pl.col(c)<0).height,"nulls":f[c].null_count()} for c in f.columns if c.endswith("_age_ms")}
    failed=not labels or duplicates or any(x["negative"] for x in ages.values()) or any(x["invalid"] or x["duplicates"] for x in labels.values())
    return {"status":"FAIL" if failed else "WARNING","rows":f.height,"duplicates":duplicates,"features":{c:stats(f,c) for c in cols},
            "high_correlation_pairs":pairs,"labels":labels,"auxiliary_ages":ages,
            "note":"Development only; correlation diagnostic, not feature selection. Assumed publication lags remain unverified."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("gate",choices=["data","features","baseline","optimization"])
    p.add_argument("--root",type=Path,default=Path("data"))
    p.add_argument("--start",default="2021-01-01")
    p.add_argument("--end",default="2025-01-01")
    a=p.parse_args()
    root=a.root
    names={"data":"01_data_quality","features":"02_feature_label","baseline":"03_predictive_benchmark","optimization":"04_optimization"}
    out=root/"reports"/(names[a.gate]+".zip")
    out.parent.mkdir(parents=True,exist_ok=True)
    files=[]
    if a.gate=="data":
        metrics=data_report(root)
        files=list((root/"metadata").rglob("*.json"))
    elif a.gate=="features":
        metrics=feature_report(root,a.start,a.end)
        files=list((root/"features"/"BTCUSDT"/"1h").glob("*.json"))
    else:
        folders=[p.parent for p in (root/"experiments").glob("*/result.json") if ("best_params" in json.loads(p.read_text()))==(a.gate=="optimization")]
        if not folders:raise ValueError("No matching completed experiments")
        metrics={"status":"WARNING","experiments":[{"id":p.name,"result":json.loads((p/"result.json").read_text())} for p in folders],"note":"Development scores, not economic or final holdout results."}
        files=[f for p in folders for f in p.rglob("*") if f.is_file() and (f.suffix==".json" or f.name=="study.sqlite3")]
    try:commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except (OSError,subprocess.CalledProcessError):commit=None
    manifest={"gate":a.gate,"git_commit":commit,"created_utc":datetime.now(timezone.utc).isoformat(),"development_start":a.start,"development_end_exclusive":a.end,"polars":pl.__version__}
    temp=out.with_suffix(".tmp")
    with zipfile.ZipFile(temp,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json",json.dumps(manifest,indent=2))
        z.writestr("metrics.json",json.dumps(metrics,indent=2,allow_nan=False))
        for f in sorted(set(files)):z.write(f,str(f.relative_to(root)))
    temp.replace(out)
    print(out,metrics["status"],flush=True)
    if metrics["status"]=="FAIL":raise SystemExit(1)
if __name__=="__main__":main()
