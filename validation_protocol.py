"""Global chronological split manifest for independent parallel folds.

All timestamps UTC milliseconds. Labels must carry actual label_end; no random
row splits. Final holdout is excluded from all development folds.
"""
from dataclasses import dataclass, asdict
from datetime import timedelta
import hashlib
import json
import polars as pl

DAY=86_400_000

@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_start: int
    test_start: int
    test_end: int

def make_folds(start, holdout_start, train_days=180, test_days=30, step_days=None):
    step_days=test_days if step_days is None else step_days
    if min(train_days,test_days,step_days)<=0 or step_days<test_days:
        raise ValueError("Positive durations and non-overlapping evaluation windows required")
    folds=[]
    cursor=start+train_days*DAY
    while cursor+test_days*DAY<=holdout_start:
        folds.append(Fold(len(folds),cursor-train_days*DAY,cursor,cursor+test_days*DAY))
        cursor+=step_days*DAY
    return folds

def select_fold(frame,fold):
    required={"decision_time","label_end","symbol"}
    if not required.issubset(frame.columns):
        raise ValueError("Missing point-in-time keys")
    if frame.select(pl.any_horizontal([pl.col(c).is_null() for c in required]).any()).item():
        raise ValueError("Null split keys")
    if frame.filter(pl.col("label_end")<=pl.col("decision_time")).height:
        raise ValueError("Label must end after decision")
    if frame.select("symbol","decision_time").unique().height!=frame.height:
        raise ValueError("Duplicate symbol/decision keys")
    t=pl.col("decision_time"); end=pl.col("label_end")
    train=frame.filter((t>=fold.train_start)&(t<fold.test_start)&(end<fold.test_start))
    test=frame.filter((t>=fold.test_start)&(t<fold.test_end)&(end<fold.test_end))
    return train.sort("decision_time","symbol"),test.sort("decision_time","symbol")

def manifest(folds,holdout_start,holdout_end,dataset_id):
    if holdout_start>=holdout_end or not dataset_id:
        raise ValueError("Invalid holdout or dataset identity")
    if any(f.test_end>holdout_start for f in folds):
        raise ValueError("Development fold crosses holdout")
    payload={"version":1,"dataset_id":dataset_id,"holdout_start":holdout_start,
             "holdout_end":holdout_end,"folds":[asdict(f) for f in folds]}
    payload["manifest_id"]=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    return payload

def causal_join(left,right,tolerance_ms):
    """Per-symbol backward as-of join on availability, never on future event time."""
    if tolerance_ms<0:raise ValueError("Negative tolerance")
    if right.filter(pl.col("available_time")<pl.col("event_time")).height:
        raise ValueError("Availability precedes observation")
    if right.select("symbol","available_time").unique().height!=right.height:
        raise ValueError("Ambiguous availability timestamp")
    out=left.sort("decision_time").join_asof(right.sort("available_time"),
        left_on="decision_time",right_on="available_time",by="symbol",
        strategy="backward",tolerance=tolerance_ms)
    return out.with_columns((pl.col("decision_time")-pl.col("available_time")).alias("data_age_ms"))
