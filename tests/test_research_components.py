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

def test_batch_wallet_shared_capacity(monkeypatch):
    import numpy as np
    import research_engine as r
    e=r.A+r.MIN; z=e+2*r.MIN
    def marks(e,z,symbol):
        return pl.DataFrame({"event_time":np.arange(e,z,r.MIN),"open":[100.] * ((z-e)//r.MIN)}),[]
    monkeypatch.setattr(r,"mark_window",marks)
    funds={}; signals=[]
    for i in range(6):
        symbol="TEST"+str(i)
        f=pl.DataFrame({"symbol":[symbol],"event_time":[r.A],"funding_rate":[0.],"mark_price":[100.]})
        funds[symbol]=(f,np.array([r.A]),np.array([0.,0.]),np.array([0,0]))
        net=r.costs(np.array([e]),np.array([z]),np.array([100.]),np.array([101.]),1,funds[symbol],5,1)[0]
        signals.append({"symbol":symbol,"decision_time":e-15*r.MIN,"entry_time":e,"label_end":z,
                        "entry_price":100.,"exit_price":101.,"side":1,"priority":float(i),
                        "fold":0,"standard":float(net),"stress":float(net)})
    out={}
    r.wallet(signals,funds,r.A,r.A+2*r.DAY,5,1,out)
    assert out["status"]=="pass"
    assert out["trades"]==5 and out["max_open"]==5
    assert out["rejected"]=={"slots":1}
    assert "TEST0" not in out["trades_per_symbol"]
    assert out["max_mtm_drawdown"]>0 and out["return_fraction"]>0
    # Future missing funding is not an entry-time rejection feature.
    signals[-1]["standard"]=None
    blocked={}
    with pytest.raises(ValueError,match="unknown funding"):
        r.wallet(signals,funds,r.A,r.A+2*r.DAY,5,1,blocked)
    assert blocked["blocked_trade"]["symbol"]=="TEST5"
