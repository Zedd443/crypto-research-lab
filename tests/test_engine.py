from datetime import datetime, timezone
import pytest
from data_engine import parse_rows, clean, gaps, aggregate, MINUTE
from universe import SYMBOLS

def sample(n=5):
    return parse_rows([[i*MINUTE,10,12,9,11,4,(i+1)*MINUTE-1,40,2,2,20] for i in range(n)])

def test_universe():
    assert len(SYMBOLS)==len(set(SYMBOLS))==37

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

def test_partial_cache_legacy(tmp_path, monkeypatch):
    import json
    import data_engine as de
    a=de.date("2020-01-01")
    b=de.date("2020-02-01")
    x,y=de.ms(a),de.ms(b)
    f=sample(4).with_columns(
        (de.pl.col("open_time")+x).alias("open_time"),
        (de.pl.col("close_time")+x).alias("close_time"))
    p=tmp_path/"futures"/"BTCUSDT"/"1m"/"2020-01.parquet"
    de.atomic_write(f,p)
    report=tmp_path/"metadata"/"BTCUSDT"/"2020-01.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"requested":[x,y],"rows":4,
        "missing_ranges_ms":[[x+4*MINUTE,y]]}))
    def no_network(*args,**kwargs):
        pytest.fail("Historical partial cache unexpectedly made a request")
    monkeypatch.setattr(de,"archive",no_network)
    monkeypatch.setattr(de,"rest",no_network)
    de.download(tmp_path,"BTCUSDT",a,b)
    assert de.pl.read_parquet(p).equals(f)
    m=json.loads(report.read_text())
    m["rows"]=5
    report.write_text(json.dumps(m))
    assert not de.cache_valid(report,p,f,x,y)
