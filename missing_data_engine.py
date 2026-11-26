"""Read-only BTC development missingness audit; no imputation or training."""
import json
import math
import zipfile
from pathlib import Path
from datetime import datetime, timezone
import polars as pl
import numpy as np

ROOT = Path('data')
HOUR = 3600000
START = int(datetime(2021,1,1,tzinfo=timezone.utc).timestamp()*1000)
END = int(datetime(2025,1,1,tzinfo=timezone.utc).timestamp()*1000)
CORE = 'ret_1 ret_4 ret_16 sma_gap_20 sma_gap_200 volatility_20 relative_volume_20 volume_delta flow_20 body range trade_size'.split()
AUX = 'sum_open_interest sum_open_interest_value count_toptrader_long_short_ratio sum_toptrader_long_short_ratio count_long_short_ratio sum_taker_long_short_vol_ratio oi_change_5m funding_rate mark_basis index_basis spot_basis premium'.split()

def runs(mask):
    edges=np.diff(np.r_[False,mask,False].astype(int))
    lengths=np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)
    return {'runs':len(lengths),'max_hours':int(lengths.max()) if len(lengths) else 0,'p95_hours':float(np.quantile(lengths,.95)) if len(lengths) else 0}

def main():
    paths=sorted((ROOT/'features'/'BTCUSDT'/'1h').glob('*.parquet'))
    if not paths: raise ValueError('No BTC feature partitions')
    f=pl.scan_parquet(paths).filter(pl.col('decision_time').is_between(START,END,closed='left')).collect().sort('decision_time')
    if not f.height: raise ValueError('Empty development period')
    absent=set(CORE+AUX)-set(f.columns)
    if absent: raise ValueError(f'Missing feature columns: {sorted(absent)}')
    t=f['decision_time'].to_numpy(); grid=np.arange(START,END,HOUR)
    if len(np.unique(t))!=len(t) or not np.isin(t,grid).all(): raise ValueError('Duplicate or off-grid decisions')
    g=pl.DataFrame({'decision_time':grid}).join(f,on='decision_time',how='left',validate='1:1')
    dates=np.array([datetime.fromtimestamp(int(x)/1000,timezone.utc).strftime('%Y-%m') for x in grid])
    obs=np.isin(grid,t); masks={}; stats={}
    vol=g['volatility_20'].cast(pl.Float64).to_numpy(); valid=np.isfinite(vol)&obs
    cutoff=float(np.quantile(vol[valid],.9)) if valid.any() else None
    high=valid&(vol>=cutoff) if cutoff is not None else valid
    low=valid&~high
    for c in CORE+AUX:
        x=g[c].cast(pl.Float64).to_numpy(); missing=~np.isfinite(x); masks[c]=missing
        stats[c]={'missing_hours':int(missing.sum()),'missing_on_present_rows':int((missing&obs).sum()),'missing_fraction':float(missing.mean()),**runs(missing),'monthly':{m:int(missing[dates==m].sum()) for m in sorted(set(dates))},'high_vol_missing_fraction':float(missing[high].mean()) if high.any() else None,'other_vol_missing_fraction':float(missing[low].mean()) if low.any() else None}
    ages={}
    for c in g.columns:
        if c.endswith('_age_ms'):
            x=g[c].cast(pl.Float64).to_numpy(); ok=np.isfinite(x)
            ages[c]={'missing':int((~ok).sum()),'negative':int((x[ok]<0).sum()),'p50_minutes':float(np.quantile(x[ok],.5)/60000) if ok.any() else None,'p95_minutes':float(np.quantile(x[ok],.95)/60000) if ok.any() else None,'max_minutes':float(x[ok].max()/60000) if ok.any() else None}
    co={a:{b:int((masks[a]&masks[b]).sum()) for b in AUX} for a in AUX}
    out={'scope':'BTC 1h development 2021-2024 only; feature-level audit, no raw source completeness or publication lag certification','expected_rows':len(grid),'actual_rows':f.height,'missing_decisions':int((~obs).sum()),'complete_core_rows':int((~np.logical_or.reduce([masks[c] for c in CORE])).sum()),'complete_all_rows':int((~np.logical_or.reduce(list(masks.values()))).sum()),'volatility_p90_descriptive_only':cutoff,'features':stats,'auxiliary_co_missing_hours':co,'freshness':ages}
    path=ROOT/'reports'/'07_missing_data.zip';path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('metrics.json',json.dumps(out,indent=2,allow_nan=False))
    tmp.replace(path);print(path,flush=True)
if __name__=='__main__':main()
