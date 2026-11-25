# Research roadmap and protocol

Goal: test net economic edge, not prove ML or manual trading superior. The 37-contract purposive pilot is not survivorship-free. Raw prices immutable; no synthetic candles or future fill.

## Progress
- [x] E00a: 1m klines, monthly Parquet, checksum, gaps, UTC aggregation.
- [x] E00b: archive inventory, derivatives metrics, funding, mark/index/premium, spot reference, 4 workers.
- [x] E00c: partial historical OHLCV cache, explicit repair, per-symbol dataset error queue; regression test added (execution to be verified).
- [x] FNG downloader, RSS collector, optional FinBERT and JSONL import (runtime not verified).
- [ ] E00d: verified listing/delisting and historical TradFi sessions, calendar-aware coverage, complete local audit and EDA, repair failed datasets.
- [ ] E00e: historical news and macro availability, trade-level data for microstructure; audit coverage before use.
- [ ] E01: point-in-time feature and label dataset, timestamp continuity, train-only preprocessing.
- [ ] E02: shared chronological validation and net execution/portfolio accounting; unit tests before model training.
- [ ] E03: technical and naive baselines; regression/classification benchmarks.
- [ ] E04: ranking/ProSP, triple barrier, derivatives/cascade, microstructure, macro, sentiment experiments.
- [ ] E05: nested tuning, ablation, parameter stability, cost and statistical robustness.
- [ ] E06: frozen holdout, forward paper evaluation, accept/reject conclusions.

## Implemented infrastructure (runtime acceptance required)
- `feature_engine.py`: source-dependent cached monthly partitions; gap-segmented technical features; bounded causal joins; separate open-to-open label tables with one full-bar execution delay. Auxiliary release lags are explicit assumptions, not verified publication timestamps. Macro/news and TradFi calendar coverage are not fabricated.
- `validation_protocol.py`: shared global chronological folds, actual-label-end purge, nonoverlapping validation windows and frozen holdout manifest.
- `experiment_store.py`: single locked Optuna coordinator, persistent SQLite study, total trial budget, interrupted-trial requeue, completed fold checkpoints, median pruning. Exact sampler trajectory after restart is not guaranteed. Mid-fit model state is deliberately not checkpointed.
- `baseline_engine.py`: Ridge, logistic, Random Forest and XGBoost development diagnostics; train-only imputation/scaling; inner chronological early stopping for XGBoost; persistent fold results. Predictive loss is not an economic benchmark. No holdout evaluation in this command.
- `resource_policy.py`, `benchmark_engine.py`: conservative RAM/CPU budget, repeated worker timings, sampled tree RSS and sequential result comparison. No four-worker Optuna SQLite concurrency.
- `execution_engine.py`: vectorized independent fixed-notional trade accounting, fee/slippage and funding from an explicitly verified ledger; fails closed when funding coverage is missing. Not a portfolio simulator.
- Regression tests cover leakage, gaps, resource budgets, cost arithmetic, cache and Optuna resume; CI installs ML dependencies.

## Acceptance commands
Run after the downloader has stopped, using the same virtual environment:
```powershell
git pull origin main
pip install -r requirements-ml.txt
python -m pytest -q
python benchmark_engine.py --partitions 12 --workers 1 2 4
python feature_engine.py --symbols BTCUSDT --tf 1h --workers 1
```
Once historical coverage has been audited, an illustrative development-only command (adjust dates to valid coverage; example is not a preapproved holdout):
```powershell
python baseline_engine.py --symbols BTCUSDT --tf 1h --horizon 4 --start 2021-01-01 --holdout-start 2025-01-01 --holdout-end 2026-01-01 --model ridge
```
Add `--trials 20` only after the baseline passes. Repeating the exact command resumes the study and completed fold calculations. A changed data/model/split identity creates a separate experiment. The example dates do not establish a genuinely unseen holdout.

## Scope exclusions
No nested four-by-four parallelism; no multi-process SQLite optimization; no mid-fit checkpointing for short CPU fits; no blind massive hyperparameter search; no artificial interpolation of raw market data. Those are deliberate exclusions, not forgotten tasks. Research experiments and full portfolio accounting remain distinct acceptance stages, not implied complete by helper modules.

## Fixed design
- Return targets: log return at 15m, 1h, 4h, 8h, 24h; entry and exit simulated using executable prices after signal availability. Direction and barrier outcomes are separate targets. No fixed 1%/5% assumption; timeout stays in evaluation. Ambiguous same-minute barrier touches handled conservatively.
- Candidate models: OLS/Ridge, logistic regression, random forest, XGBoost; LightGBM for microstructure risk, VAR for macro, EVT for tail risk, FinBERT for sentiment. No stacking, deep learning, RL, Kelly sizing, or automatic execution in this scope.
- Features: return/momentum, volatility, volume/OBV, liquidity, trend, candle geometry, OI, funding, basis/premium, aggressor flow, cross-sectional ranks, external macro/news/FNG. Remove exact redundancy; test correlated groups via fold-local ablation. VPIN/Roll require suitable trade records; never label OHLCV proxies as exact measures.
- Signed IC is diagnostic, not strategy P&L. Determine direction on training data only; no forced positive correlation. Parameter stability: compare neighborhood standard deviation (center included, insufficient neighbors invalid) on profitable in-sample strategies against best-point selection, then evaluate both on untouched next window. No reuse of old IDW implementation.
- Global timestamp splits for all symbols; purge by actual label end, nested selection, frozen final holdout. All scaling/selection fit inside training folds. Record data availability separately from event time. No zero-fill for missing datasets.
- Backtest next executable price, configurable historical fee schedule, funding, slippage, turnover, exposure, realistic fill assumptions. Cash, naive and exposure-matched benchmarks; fixed unlevered sizing baseline. No claim to reproduce liquidation/ADL without required historical inputs.
- Report net expectancy, equity, drawdown, risk-adjusted return, calibration, OOS IC, turnover, fold stability, block bootstrap, multiple-testing adjustment, cost stress. Negative results retained. No guaranteed profit.
- Fail on leakage, wrong class masks, missing required features, price mutations, wrong optimization direction, schema mismatches, or fabricated metrics. Missing independent data sources are explicitly marked unavailable, not silently replaced.

## References
- Lin et al. (2024): https://doi.org/10.3390/engproc2024074056 (parameter stability, not evidence of crypto edge).
- Algorithmic crypto trading using information-driven bars and triple barrier: https://doi.org/10.1186/s40854-025-00866-w (testable hypothesis, not a guarantee).
- Prior uploaded empirical literature and research decisions remain evidence inputs; benchmark all candidates on actual Binance data.

Local data completeness and CI status must be verified separately from code commits.

## Validation bundles and BTC pilot
Run after the download stops; review each ZIP before the next gate. Reports are diagnostic, not a certificate of data quality. Existing raw partitions are not changed.
```cmd
python research_pipeline.py audit --workers 4
python research_pipeline.py compare --symbols BTCUSDT --start 2024-01-01 --end 2024-02-01
python report_engine.py data
python news_engine.py fng
python benchmark_engine.py --partitions 24 --workers 1 2 4 --repeats 3
python feature_engine.py --symbols BTCUSDT --tf 1h --workers 1
python report_engine.py features --start 2021-01-01 --end 2025-01-01
```
Pilot development (holdout reserved, not certified unseen):
```cmd
for %M in (ridge logistic rf xgb) do python baseline_engine.py --symbols BTCUSDT --tf 1h --horizon 4 --start 2021-01-01 --holdout-start 2025-01-01 --holdout-end 2026-01-01 --train-days 180 --test-days 30 --threads 2 --seed 42 --model %M
python report_engine.py baseline
```
After review only, example tuning run (20 total trials, not 20 additional on resume):
```cmd
python baseline_engine.py --symbols BTCUSDT --tf 1h --horizon 4 --start 2021-01-01 --holdout-start 2025-01-01 --holdout-end 2026-01-01 --train-days 180 --test-days 30 --threads 2 --seed 42 --model ridge --trials 20
python report_engine.py optimization
```
Bundles under data/reports: 01_data_quality.zip, 02_feature_label.zip, 03_predictive_benchmark.zip, 04_optimization.zip. Benchmark separately data/metadata/compute_benchmark.json. Feature report covers development only; experiment bundles include all matching completed experiments, not an isolated run. No full backtest/holdout command yet. No verified listing/calendar, OOS prediction export, full ablation, or portfolio report yet.
