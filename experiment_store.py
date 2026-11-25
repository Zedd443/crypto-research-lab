"""Persistent fold checkpoints and resumable, single-coordinator Optuna tuning."""
import hashlib
import json
import math
import os
from pathlib import Path
import optuna
from optuna.trial import TrialState
from optuna.storages import RDBStorage
from data_engine import write_json

def identity(spec):
    return hashlib.sha256(json.dumps(spec,sort_keys=True,allow_nan=False).encode()).hexdigest()

def checkpoint(root,spec,fold_id,compute):
    """Reuse only identical data/split/model/parameter/seed spec."""
    key=identity(spec)
    p=Path(root)/key/(str(fold_id)+".json")
    if p.exists():
        record=json.loads(p.read_text())
        if record["identity"]!=key:raise ValueError("Checkpoint identity mismatch")
        return record["result"]
    result=compute()
    json.dumps(result,allow_nan=False)
    write_json(p,{"identity":key,"result":result})
    return result

def tune(root,spec,objective,total_trials=50,seed=42):
    """One coordinator per study; model internals may use bounded threads.
    Abandoned RUNNING trials are marked FAIL on resume, not counted as completed.
    Do not run two processes against the same study directory.
    """
    if total_trials<1:raise ValueError("Trial budget must be positive")
    root=Path(root)/identity(spec)
    root.mkdir(parents=True,exist_ok=True)
    # OS releases advisory file lock automatically after process exit/restart.
    import portalocker
    with portalocker.Lock(str(root/"coordinator.lock"),timeout=0):
        storage=RDBStorage("sqlite:///"+(root/"study.sqlite3").resolve().as_posix())
        study=optuna.create_study(study_name=identity(spec),storage=storage,load_if_exists=True,
            direction="maximize",sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5,n_warmup_steps=3))
        for t in study.get_trials(deepcopy=False,states=(TrialState.RUNNING,)):
            study.tell(t.number,state=TrialState.FAIL)
        done=len(study.get_trials(deepcopy=False,states=(TrialState.COMPLETE,TrialState.PRUNED)))
        if done<total_trials:
            study.optimize(objective,n_trials=total_trials-done,n_jobs=1,gc_after_trial=True)
        return study

def report_fold(trial,step,score):
    if not math.isfinite(score):raise ValueError("Nonfinite objective")
    trial.report(score,step)
    if trial.should_prune():raise optuna.TrialPruned()
