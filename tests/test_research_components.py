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
    assert score["score"]==pytest.approx(.02/11)
    missing={**signals[0],"standard":None,"stress":None}
    result=edge_stats([missing,signals[1]],"standard",A,A+DAY)
    assert result["signals"]==2 and result["scored_signals"]==1
    assert result["unknown_funding_signals"]==1 and result["coverage"]==.5
    assert inner_score([missing],A,A+DAY)["score"] is None


def test_batch_primary_edge_not_vetoed_by_stress():
    from research_engine import inner_score,A,DAY
    t={"symbol":"BTC","decision_time":A,"entry_time":A+60000,
       "label_end":A+3600000,"standard":.002,"stress":-.001}
    r=inner_score([t],A,A+DAY)
    assert r["score"]==pytest.approx(.002/11)
    assert r["stress_expectancy"]<0

def test_batch_inner_masks_purge_common_horizon_and_missing_funding():
    import numpy as np
    import research_engine as r
    from validation_protocol import Fold
    fold=Fold(0,r.A,r.A+180*r.DAY,r.A+210*r.DAY)
    d=np.arange(r.A,fold.test_end,r.H)
    ds={"d":d,"e":d+15*r.MIN,"z":d+15*r.MIN+4*r.H,
        "common_known":np.ones(len(d),dtype=bool)}
    ds["common_known"][120*24+1]=False
    windows=list(r.inner_masks(ds,fold))
    assert len(windows)==2
    for train,val,lo,hi,total in windows:
        assert ds["z"][train].max()<lo
        assert ds["d"][val].min()>=lo
        assert (ds["e"][val]+24*r.H).max()<hi
        assert hi<=fold.test_start
        assert ds["common_known"][val].all()
        assert total>=val.sum()

def test_batch_rule_cutoff_is_train_only():
    import numpy as np
    import research_engine as r
    x=np.zeros((160,len(r.COLS)))
    k=r.COLS.index("ret_4")
    x[:,k]=np.linspace(-1,1,160)
    train=np.arange(160)<120;test=~train
    a=r.rule_signals({"x":x},train,test,"ret_4",.8,"momentum")
    changed=x.copy();changed[test,k]*=2
    b=r.rule_signals({"x":changed},train,test,"ret_4",.8,"momentum")
    cutoff=np.quantile(np.abs(x[train,k]),.8)
    expected=np.where(np.abs(changed[test,k])>cutoff,np.sign(changed[test,k]),0)
    assert np.array_equal(b,expected)
    assert np.array_equal(-a,r.rule_signals({"x":x},train,test,"ret_4",.8,"reversal"))

def test_batch_vectorized_inner_matches_signal_ledger():
    import numpy as np
    import research_engine as r
    d=r.A+np.arange(8)*r.H;e=d+15*r.MIN;z=e+2*r.H
    long=(z,np.full(8,101.),np.linspace(-.01,.03,8),np.linspace(-.02,.02,8))
    short=(z,np.full(8,99.),-long[2],-long[3])
    ds={"d":d,"e":e,"ep":np.full(8,100.),"variants":{(1,"fixed"):long,(-1,"fixed"):short}}
    test=np.ones(8,dtype=bool);p=(np.array([.2,.8,.2,.8,.2,.8,.2,.8]),np.array([.8,.2,.8,.2,.8,.2,.8,.2]))
    rows=r.opportunities(ds,test,p,.7,"fixed")
    n,a,b=r.inner_outcomes(ds,test,p,.7,"fixed")
    assert r.summarize_inner(n,a,b)==r.inner_score(rows,r.A,r.A+r.DAY)
    assert len(rows)==8 and n==8

def test_batch_matched_direction_control():
    from research_engine import edge_stats,A,DAY
    rows=[{"standard":.01,"opposite_standard":-.03},{"standard":.03,"opposite_standard":-.01}]
    result=edge_stats(rows,"standard",A,A+DAY)
    assert result["expectancy"]==pytest.approx(.02)
    assert result["matched_direction_comparator"]["chosen_minus_equal_random_direction_expectancy"]==pytest.approx(.02)


def test_transfer_retains_unknown_price_and_funding():
    import numpy as np
    import research_engine as r
    f=pl.DataFrame({"symbol":["X","X"],"decision_time":[r.A,r.A+r.H],
                    "entry_time":[r.A+15*r.MIN,r.A+r.H+15*r.MIN],
                    "label_end":[r.A+4*r.H+15*r.MIN,r.A+5*r.H+15*r.MIN],
                    "entry_price":[100.,100.],"exit_price":[None,101.]})
    rows=r.transfer_extract(f,np.array([.8,.8]),0)
    assert len(rows)==2
    r.transfer_score(rows,None,[])
    assert rows[0]["outcome_status"]=="unknown_price"
    assert rows[0]["fee_slip_standard"] is None
    assert rows[1]["outcome_status"]=="unverified_funding_schedule"
    assert rows[1]["standard"] is None and rows[1]["fee_slip_standard"]>0

def test_transfer_local_funding_spans_do_not_bridge_gap(tmp_path,monkeypatch):
    import research_engine as r
    monkeypatch.setattr(r,"ROOT",tmp_path)
    ts=[r.A,r.A+8*r.H,r.A+24*r.H,r.A+32*r.H]
    p=tmp_path/"funding"/"X"/"2021-01.parquet"
    p.parent.mkdir(parents=True)
    pl.DataFrame({"event_time":ts,"funding_rate":[.001]*4,
                  "funding_interval_hours":[8]*4}).write_parquet(p)
    q=tmp_path/"mark"/"X"/"2021-01.parquet"
    q.parent.mkdir(parents=True)
    pl.DataFrame({"event_time":ts,"open":[100.]*4}).write_parquet(q)
    fund,spans=r.transfer_funding("X")
    assert spans==[(ts[0],ts[1]),(ts[2],ts[3])]
    rows=[{"symbol":"X","entry_time":r.A+r.H,"label_end":r.A+5*r.H,
           "entry_price":100.,"exit_price":101.,"side":1},
          {"symbol":"X","entry_time":r.A+7*r.H,"label_end":r.A+11*r.H,
           "entry_price":100.,"exit_price":101.,"side":1}]
    r.transfer_score(rows,fund,spans)
    assert rows[0]["outcome_status"]=="scored"
    assert rows[1]["outcome_status"]=="unverified_funding_schedule"
    assert rows[1]["standard"] is None


def test_transfer_temporal_funding_uses_requested_calendar(tmp_path,monkeypatch):
    import research_engine as r
    from data_engine import date,ms
    import numpy as np
    monkeypatch.setattr(r,"ROOT",tmp_path)
    a,b=ms(date("2025-01-01")),ms(date("2025-01-02"))
    ts=np.arange(a,b,8*r.H).tolist()
    p=tmp_path/"funding"/"BTCUSDT"/"2025-01.parquet"
    p.parent.mkdir(parents=True)
    pl.DataFrame({"event_time":ts,"funding_rate":[.001]*3,
                  "funding_interval_hours":[8]*3}).write_parquet(p)
    q=tmp_path/"mark"/"BTCUSDT"/"2025-01.parquet"
    q.parent.mkdir(parents=True)
    pl.DataFrame({"event_time":ts,"open":[100.]*3}).write_parquet(q)
    fund,spans=r.transfer_funding("BTCUSDT",a,b)
    assert spans==[(a,b)]
    assert fund[0].height==3
    # A trade after the last verified event is covered only within the complete requested grid.
    rows=[{"symbol":"BTCUSDT","entry_time":a+18*r.H,"label_end":a+22*r.H,
           "entry_price":100.,"exit_price":101.,"side":1}]
    r.transfer_score(rows,fund,spans)
    assert rows[0]["outcome_status"]=="scored"
