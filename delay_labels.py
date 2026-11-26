"""Build development-only 15m latency labels, preserving original labels."""
from pathlib import Path
from datetime import timedelta
import polars as pl
import data_engine as de


def main():
    root=Path('data'); symbol='BTCUSDT'
    a=de.ms(de.date('2021-01-01')); b=de.ms(de.date('2025-01-01'))
    raw=sorted((root/'futures'/symbol/'1m').glob('*.parquet'))
    features=sorted((root/'features'/symbol/'1h').glob('*.parquet'))
    if not raw or not features:
        raise ValueError('Missing raw prices or features')
    for p in features:
        if not '2021-01'<=p.stem<='2024-12': continue
        f=pl.read_parquet(p,columns=['symbol','decision_time']).filter(pl.col('decision_time').is_between(a,b,closed='left'))
        if not f.height: continue
        if f['decision_time'].n_unique()!=f.height: raise ValueError('Duplicate decision times')
        lo=f['decision_time'].min(); hi=min(b,f['decision_time'].max()+25*3600000)
        paths=[]
        for q in raw:
            month=de.date(q.stem+'-01')
            nxt=(month+timedelta(days=32)).replace(day=1)
            if de.ms(month)<hi and de.ms(nxt)>lo: paths.append(q)
        if not paths: raise ValueError('Missing raw partition '+p.stem)
        prices=pl.scan_parquet(paths).select('open_time','open').filter(pl.col('open_time').is_between(lo,hi,closed='left')).collect()
        if prices['open_time'].n_unique()!=prices.height: raise ValueError('Duplicate minute prices')
        for h in (1,4,8,24):
            t=f.with_columns((pl.col('decision_time')+900000).alias('entry_time'),(pl.col('decision_time')+900000+h*3600000).alias('label_end')).filter(pl.col('label_end')<b)
            expected=t.height
            for key,name in (('entry_time','entry_price'),('label_end','exit_price')):
                t=t.join(prices.rename({'open_time':key,'open':name}),on=key,how='left',validate='m:1')
            t=t.filter(pl.all_horizontal([pl.col(c).is_finite() & (pl.col(c)>0) for c in ('entry_price','exit_price')])).with_columns((pl.col('exit_price')/pl.col('entry_price')).log().alias('target_return')).sort('decision_time')
            target=root/'labels_delay_15'/symbol/'1h'/str(h)/p.name
            de.atomic_write(t,target)
            print(p.stem,h,t.height,'labels; missing',expected-t.height,flush=True)
if __name__=='__main__': main()
