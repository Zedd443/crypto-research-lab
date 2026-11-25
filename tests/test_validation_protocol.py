import polars as pl
import pytest
import data_engine as de
from validation_protocol import Fold, make_folds, select_fold, manifest, causal_join, DAY

def test_purge_and_global_split():
    f=pl.DataFrame({"symbol":["BTC","ETH","BTC","ETH","BTC"],"decision_time":[1,1,4,5,8],"label_end":[2,5,6,6,10]})
    train,test=select_fold(f,Fold(0,0,5,10))
    assert train["decision_time"].to_list()==[1]
    assert test["decision_time"].to_list()==[5]

def test_fold_manifest():
    folds=make_folds(0,365*DAY,180,30)
    assert len(folds)==6
    assert all(f.test_end<=365*DAY for f in folds)
    assert manifest(folds,365*DAY,400*DAY,"v1")==manifest(folds,365*DAY,400*DAY,"v1")
    with pytest.raises(ValueError):
        make_folds(0,365*DAY,180,30,15)

def test_causal_join():
    left=pl.DataFrame({"symbol":["BTC","BTC","ETH"],"decision_time":[10,20,20]})
    right=pl.DataFrame({"symbol":["BTC","BTC"],"event_time":[5,9],"available_time":[8,15],"oi":[100,200]})
    out=causal_join(left,right,10)
    assert out.filter(pl.col("symbol")=="BTC")["oi"].to_list()==[100,200]
    assert out.filter(pl.col("symbol")=="ETH")["oi"].null_count()==1

def test_vectorized_gap():
    f=pl.DataFrame({"open_time":[60000,120000,240000]})
    assert list(de.gaps(f,0,360000))==[(0,60000),(180000,240000),(300000,360000)]

def test_resample_cache(tmp_path,monkeypatch):
    a=de.ms(de.date("2024-01-01"))
    f=de.parse_rows([[a+i*60000,10,12,9,11,4,a+(i+1)*60000-1,40,2,2,20] for i in range(15)])
    p=tmp_path/"futures"/"BTCUSDT"/"1m"/"2024-01.parquet"
    de.atomic_write(f,p)
    de.resample(tmp_path,"BTCUSDT")
    original=de.aggregate
    def unexpected(*args,**kwargs):
        pytest.fail("Unchanged resample should be cached")
    monkeypatch.setattr(de,"aggregate",unexpected)
    de.resample(tmp_path,"BTCUSDT")
    monkeypatch.setattr(de,"aggregate",original)
    target=tmp_path/"resampled"/"BTCUSDT"/"15m"/p.name
    target.unlink()
    de.resample(tmp_path,"BTCUSDT")
    assert pl.read_parquet(target).height==1
