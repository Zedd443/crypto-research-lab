"""Paired BTC execution-delay screening using saved predictions."""
import json, zipfile
from pathlib import Path
import numpy as np
import polars as pl
from robustness_engine import ledger, pick, A, B
from execution_engine import score_trades
from experiment_store import identity

DELAYS = (1, 5, 15, 60)

def main():
    root = Path('data')
    fund = ledger(root)
    missing = fund.filter(pl.col('mark_price').is_null())['event_time'].to_numpy()
    fund = fund.filter(pl.col('mark_price').is_not_null())
    raw = [p for p in sorted((root/'futures'/'BTCUSDT'/'1m').glob('*.parquet')) if '2021-01' <= p.stem <= '2024-12']
    if not raw: raise ValueError('Missing raw BTC prices')
    results = []
    for p in sorted((root/'experiments').glob('*/result.json')):
        folder = p.parent
        s = json.loads((folder/'spec.json').read_text())
        if not s.get('prediction_export') or s['model'] not in ('logistic','xgb') or s.get('delay_minutes',60)!=60: continue
        preds = sorted((folder/'predictions').glob('*.parquet'))
        key = identity({**s, 'params':json.loads(p.read_text())['params']})
        checks = list((folder/'folds'/key).glob('*.json'))
        if not checks or {x.stem for x in checks} != {x.stem for x in preds}: raise ValueError('Incomplete exports')
        for c in checks:
            if pl.read_parquet(folder/'predictions'/(c.stem+'.parquet')).height != json.loads(c.read_text())['result']['test_rows']: raise ValueError('Row mismatch')
        f = pl.read_parquet(preds).sort('decision_time')
        if f['symbol'].unique().to_list() != ['BTCUSDT'] or f['decision_time'].n_unique()!=f.height or f['decision_time'].min()<A or f['label_end'].max()>=B: raise ValueError('Invalid scope')
        if not np.isfinite(f['prediction'].to_numpy()).all(): raise ValueError('Invalid prediction')
        h = s['horizon']; g = f
        times = np.unique(np.concatenate([f['decision_time'].to_numpy()+k*60000 for d in DELAYS for k in (d,d+60*h)]))
        prices = pl.scan_parquet(raw).select('open_time','open').filter(pl.col('open_time').is_in(pl.Series(times).implode())).collect()
        for d in DELAYS:
            for name,k in (('entry',d),('exit',d+60*h)):
                col = name+'_time_'+str(d)
                g = g.with_columns((pl.col('decision_time')+k*60000).alias(col)).join(prices.rename({'open_time':col,'open':name+'_price_'+str(d)}),on=col,how='left',validate='m:1')
        g = g.filter(pl.all_horizontal([pl.col(n+'_price_'+str(d)).is_not_null() & pl.col(n+'_price_'+str(d)).is_finite() & (pl.col(n+'_price_'+str(d))>0) for d in DELAYS for n in ('entry','exit')]))
        if g.filter((pl.col('entry_time')!=pl.col('entry_time_60'))|(pl.col('label_end')!=pl.col('exit_time_60'))|((pl.col('entry_price')/pl.col('entry_price_60')-1).abs()>1e-9)|((pl.col('exit_price')/pl.col('exit_price_60')-1).abs()>1e-9)).height: raise ValueError('60m baseline mismatch')
        rows = []
        for th in ([.5,.52,.55,.6,.65,.7] if s['model']=='logistic' else [0,.0005,.001,.002,.005]):
            original = pick(f,s['model'],th)
            t = original.select('decision_time','side').join(g,on='decision_time',how='inner',validate='1:1').sort('decision_time')
            eligible = np.ones(t.height,dtype=bool)
            for d in DELAYS:
                entry=t['entry_time_'+str(d)].to_numpy(); end=t['exit_time_'+str(d)].to_numpy()
                eligible &= np.searchsorted(missing,end,side='right')==np.searchsorted(missing,entry,side='right')
            price_count=t.height; t=t.filter(pl.Series(eligible))
            for d in DELAYS:
                u=t.select('symbol','decision_time','side',pl.col('entry_time_'+str(d)).alias('entry_time'),pl.col('exit_time_'+str(d)).alias('label_end'),pl.col('entry_price_'+str(d)).alias('entry_price'),pl.col('exit_price_'+str(d)).alias('exit_price'))
                for fee,slip in [(2,0),(5,1),(5,3),(10,5)]:
                    row={'threshold':th,'delay_minutes':d,'fee_bps':fee,'slippage_bps':slip,'original_trades':original.height,'excluded_prices':original.height-price_count,'excluded_funding':price_count-t.height,'trades':t.height}
                    if t.height:
                        scored=score_trades(u,fee,slip,fund,{'BTCUSDT':[(A,B)]})
                        r=scored['net_return'].to_numpy(); loss=-r[r<0].sum()
                        row.update(expectancy=float(r.mean()),pnl=float(r.sum()),win_rate=float((r>0).mean()),profit_factor=float(r[r>0].sum()/loss) if loss else None,funding=float(scored['funding_return'].sum()))
                    rows.append(row)
        results.append({'id':folder.name,'spec':s,'rows':rows}); print(s['model'],h,'done',flush=True)
    if not results: raise ValueError('No exported experiments')
    out=root/'reports'/'06_execution_delay.zip'; out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('metrics.json',json.dumps({'note':'Paired fixed original signals, common eligible trades, original 60m schedule; no retraining. Funding mark proxy and assumed costs. Development only.','experiments':results},allow_nan=False))
    tmp.replace(out); print(out)
if __name__=='__main__': main()
