import polars as pl
import pytest
import data_engine as de
from feature_engine import technical, labels, attach
from execution_engine import score_trades
from experiment_store import checkpoint
from resource_policy import budget

STEP=900000

def candles(n=220):
    return de.parse_rows([[i*STEP,10+i,12+i,9+i,11+i,4,(i+1)*STEP-1,40,2,2,20] for i in range(n)])

def test_gap_resets_features_and_invalidates_label():
    f=candles().filter(pl.col("open_time")!=201*STEP)
    out=technical(f,STEP)
    assert out.filter(pl.col("open_time")==202*STEP)["ret_1"][0] is None
    l=labels(f,STEP,1)
    assert l.filter(pl.col("decision_time")==200*STEP).is_empty()

def test_future_prices_do_not_change_features():
    f=candles()
    original=technical(f,STEP).filter(pl.col("open_time")<210*STEP)
    changed=f.with_columns(pl.when(pl.col("open_time")>=210*STEP).then(pl.col("close")*2).otherwise(pl.col("close")).alias("close"))
    assert original.equals(technical(changed,STEP).filter(pl.col("open_time")<210*STEP))

def test_aux_lag_and_staleness():
    f=pl.DataFrame({"decision_time":[299999,300000,600000,600001]})
    r=pl.DataFrame({"event_time":[0],"oi":[10.]})
    out=attach(f,r,["oi"],300000,300000,"metrics")
    assert out["oi"].to_list()==[None,10.,10.,None]

def test_checkpoint(tmp_path):
    spec={"dataset":"v1","fold":"a","parameter":1}
    assert checkpoint(tmp_path,spec,0,lambda:{"score":2})=={"score":2}
    def fail():pytest.fail("Checkpoint should be reused")
    assert checkpoint(tmp_path,spec,0,fail)=={"score":2}
    assert checkpoint(tmp_path,{**spec,"parameter":2},0,lambda:{"score":3})=={"score":3}

def test_funding_and_costs():
    t=pl.DataFrame({"symbol":["BTC","BTC"],"entry_time":[100,100],"label_end":[200,200],
        "entry_price":[100.,100.],"exit_price":[110.,110.],"side":[1,-1]})
    f=pl.DataFrame({"symbol":["BTC"],"event_time":[150],"funding_rate":[.001],"mark_price":[105.]})
    s=score_trades(t,fee_bps=5,slippage_bps=0,funding=f,coverage={"BTC":[(0,300)]})
    assert s["net_return"][0]==pytest.approx(.1-.00105-.00105)
    assert s["net_return"][1]==pytest.approx(-.1-.00105+.00105)
    with pytest.raises(ValueError):
        score_trades(t,funding=f,coverage={})

def test_budget(monkeypatch):
    import resource_policy as r
    monkeypatch.setattr(r.psutil,"virtual_memory",lambda:type("M",(),{"available":3*r.GIB})())
    monkeypatch.setattr(r.os,"cpu_count",lambda:4)
    assert budget(4,r.GIB,r.GIB)==2
