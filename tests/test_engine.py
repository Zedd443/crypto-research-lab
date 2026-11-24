from datetime import datetime, timezone
import pytest
from data_engine import parse_rows, clean, gaps, aggregate, MINUTE
from universe import SYMBOLS

def sample(n=5):
    return parse_rows([[i*MINUTE,10,12,9,11,4,(i+1)*MINUTE-1,40,2,2,20] for i in range(n)])

def test_universe():
    assert len(SYMBOLS)==len(set(SYMBOLS))==30

def test_resample():
    f=aggregate(clean(sample(),0,5*MINUTE),"5m")
    assert f.height==1
    assert f["volume"][0]==20
    assert f["trades"][0]==10
    assert f["high"][0]==12

def test_missing():
    f=sample(4)
    assert list(gaps(f,0,5*MINUTE))==[(4*MINUTE,5*MINUTE)]
    assert aggregate(f,"5m").is_empty()

def test_duplicate_conflict():
    import polars as pl
    f=sample()
    with pytest.raises(ValueError):
        clean(pl.concat([f,f.head(1).with_columns(pl.lit(10.5).alias("close"))]),0,5*MINUTE)
