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

def test_optuna_resume_budget(tmp_path):
    from experiment_store import tune
    def objective(trial):
        return trial.suggest_float("x",0,1)
    first=tune(tmp_path,{"fixture":"resume"},objective,2)
    assert len(first.trials)==2
    second=tune(tmp_path,{"fixture":"resume"},objective,3)
    assert len(second.trials)==3
    third=tune(tmp_path,{"fixture":"resume"},objective,3)
    assert len(third.trials)==3

def test_ridge_baseline():
    from baseline_engine import fit_score
    from feature_engine import BASE_FEATURES
    f=pl.DataFrame({c:[float(i%17) for i in range(220)] for c in BASE_FEATURES}).with_columns(
        pl.Series("target_return",[(i%17)*.0001 for i in range(220)]))
    r=fit_score(f.head(180),f.tail(40),BASE_FEATURES,"ridge",{"alpha":1.},1,42)
    assert r["loss"]<r["baseline_loss"]
    assert r["test_rows"]==40


def test_batch_funding_boundary_and_unknown():
    import numpy as np
    from research_engine import costs
    f=pl.DataFrame({"symbol":["BTC","BTC"],"event_time":[100,150],"funding_rate":[.002,.001],"mark_price":[100.,105.]})
    t=pl.DataFrame({"symbol":["BTC","BTC"],"entry_time":[100,100],"label_end":[150,150],
        "entry_price":[100.,100.],"exit_price":[110.,110.],"side":[1,-1]})
    fund=(f,np.array([100,150]),np.array([0.,.2,.305]),np.array([0,0,0]))
    args=[t[c].to_numpy() for c in ["entry_time","label_end","entry_price","exit_price","side"]]
    expected=score_trades(t,5,1,f,{"BTC":[(0,300)]})["net_return"].to_numpy()
    assert np.allclose(costs(*args,fund,5,1),expected)
    bad=(f,fund[1],fund[2],np.array([0,0,1]))
    assert np.isnan(costs(*args,bad,5,1)).all()

def test_batch_signal_edge_no_shared_capital():
    from research_engine import nonoverlap,inner_score,edge_stats,A,DAY
    e=A+60000;z=e+3600000
    signals=[{"symbol":"TEST"+str(i),"decision_time":A,"entry_time":e,"label_end":z,
              "side":1,"standard":.02,"stress":.01,"fold":0} for i in range(100)]
    # All 100 simultaneous symbols count, regardless of any notional budget.
    assert len(nonoverlap(signals))==100
    # A repeated opportunity in one coin remains in the raw signal sample.
    repeated={**signals[0],"decision_time":e,"entry_time":e+60000}
    all_signals=signals+[repeated]
    assert len(nonoverlap(all_signals))==100
    assert edge_stats(all_signals,"standard",A,A+DAY)["signals"]==101
    # Sparse positive evidence is eligible, but shrunk rather than forced to trade.
    score=inner_score(signals[:1],A,A+DAY)
    assert score["score"]==pytest.approx(.01/11)
    missing={**signals[0],"standard":None,"stress":None}
    result=edge_stats([missing,signals[1]],"standard",A,A+DAY)
    assert result["signals"]==2 and result["scored_signals"]==1
    assert result["unknown_funding_signals"]==1 and result["coverage"]==.5
    assert inner_score([missing],A,A+DAY)["score"] is None
