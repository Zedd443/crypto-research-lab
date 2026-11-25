"""Reference execution on prevalidated open-to-open labels.
Independent trades, not a portfolio simulator. No liquidation or fill claims.
"""
import polars as pl

def score_trades(trades,fee_bps=5.0,slippage_bps=1.0,funding=None,coverage=None):
    required={"symbol","entry_time","label_end","entry_price","exit_price","side"}
    if not required.issubset(trades.columns):raise ValueError("Missing trade fields")
    if not 0<=fee_bps<10000 or not 0<=slippage_bps<10000:raise ValueError("Invalid costs")
    if trades.filter(pl.any_horizontal([pl.col(c).is_null() for c in required]) |
        ~pl.col("side").is_in([-1,1]) | (pl.col("label_end")<=pl.col("entry_time")) |
        (pl.col("entry_price")<=0)|(pl.col("exit_price")<=0) |
        ~pl.col("entry_price").is_finite()|~pl.col("exit_price").is_finite()).height:
        raise ValueError("Invalid trade")
    if funding is None or coverage is None:
        raise ValueError("Funding data and independently verified coverage required")
    # coverage: explicit verified intervals per contract; must cover entire trade.
    for symbol,entry,end in trades.select("symbol","entry_time","label_end").iter_rows():
        spans=coverage.get(symbol,[])
        if not any(a<=entry and b>=end for a,b in spans):
            raise ValueError("Unverified funding coverage: "+symbol)
    required_f={"symbol","event_time","funding_rate","mark_price"}
    if not required_f.issubset(funding.columns):raise ValueError("Invalid funding ledger")
    if funding.select("symbol","event_time").unique().height!=funding.height:
        raise ValueError("Duplicate funding events")
    if funding.filter(pl.any_horizontal([pl.col(c).is_null() for c in required_f]) |
        ~pl.col("funding_rate").is_finite() | ~pl.col("mark_price").is_finite() |
        (pl.col("mark_price")<=0)).height:raise ValueError("Invalid funding event")
    ledger=funding.sort("symbol","event_time").with_columns(
        (pl.col("funding_rate")*pl.col("mark_price")).cum_sum().over("symbol").alias("cum"))
    t=trades.with_row_index("_trade_id")
    for key,name in [("entry_time","funding_before"),("label_end","funding_after")]:
        t=t.sort(key).join_asof(ledger.select("symbol","event_time",pl.col("cum").alias(name)).sort("event_time"),
             left_on=key,right_on="event_time",by="symbol",strategy="backward").drop("event_time")
    t=t.with_columns((pl.col("side")*(pl.col("funding_after").fill_null(0)-pl.col("funding_before").fill_null(0))).alias("funding_cash_per_unit"))
    side=pl.col("side"); slip=slippage_bps/10000; fee=fee_bps/10000
    t=t.with_columns((pl.col("entry_price")*(1+side*slip)).alias("entry_fill"),
                     (pl.col("exit_price")*(1-side*slip)).alias("exit_fill"))
    return t.with_columns((side*(pl.col("exit_fill")-pl.col("entry_fill"))/pl.col("entry_fill")).alias("gross_return"),
          (fee*(pl.col("entry_fill")+pl.col("exit_fill"))/pl.col("entry_fill")).alias("fee_return"),
          (pl.col("funding_cash_per_unit")/pl.col("entry_fill")).alias("funding_return")).with_columns(
          (pl.col("gross_return")-pl.col("fee_return")-pl.col("funding_return")).alias("net_return")).sort("_trade_id").drop("_trade_id")

def summary(scored):
    if not scored.height:raise ValueError("No trades")
    r=scored["net_return"]
    wins=r.filter(r>0).sum()
    losses=-r.filter(r<0).sum()
    return {"trades":len(r),"mean_net_return":r.mean(),"win_rate":float((r>0).mean()),
            "profit_factor":wins/losses if losses else None,
            "note":"Independent fixed-notional trades; not compounded portfolio equity"}
