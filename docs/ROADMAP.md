# Research roadmap and protocol

Goal: test net economic edge, not prove ML or manual trading superior. The 37-contract purposive pilot is not survivorship-free. Raw prices immutable; no synthetic candles or future fill.

## Progress
- [x] E00a: 1m klines, monthly Parquet, checksum, gaps, UTC aggregation.
- [x] E00b: archive inventory, derivatives metrics, funding, mark/index/premium, spot reference, 4 workers.
- [x] E00c: partial historical OHLCV cache, explicit repair, per-symbol dataset error queue; regression test added; historical local suite passed.
- [x] FNG downloader, RSS collector, optional FinBERT and JSONL import (runtime not verified).
- [ ] E00d: verified listing/delisting and historical TradFi sessions, calendar-aware coverage, complete local audit and EDA, repair failed datasets.
- [ ] E00e: historical news and macro availability, trade-level data for microstructure; audit coverage before use.
- [ ] E01: point-in-time feature and label dataset, timestamp continuity, train-only preprocessing.
- [ ] E02: shared chronological validation and net execution/portfolio accounting; unit tests before model training.
- [x] E03a: BTC hourly technical and naive predictive baselines; 1/4/8/24h horizons; initial derivatives comparison, development only.
- [ ] E03b: rule-based trading baselines and predictive diagnostics on revised labels.
- [x] E04a: BTC fixed-signal execution-delay comparison, 15m relabel/retrain, economic screening, 12 leave-one-out ablations and feature missingness audit (local reports).
- [ ] E04b: reduced derivatives, controlled missingness experiments, revised features and targets, ranking/ProSP, triple barrier, cascade, microstructure, macro and sentiment.
- [ ] E05: integrity audit, nested selection, parameter stability, portfolio accounting, cross-asset validation and multiple-testing-adjusted robustness.
- [ ] E06: frozen holdout, forward paper evaluation, accept/reject conclusions.

## Implemented infrastructure (runtime acceptance required)
- `feature_engine.py`: source-dependent cached monthly partitions; gap-segmented technical features; bounded causal joins; separate open-to-open label tables; original full-bar delay and isolated 15-minute BTC hourly pilot labels. Auxiliary release lags are explicit assumptions, not verified publication timestamps. Macro/news and TradFi calendar coverage are not fabricated.
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

## Research checkpoint — 2026-11-26

Source: user-provided local validation bundles; raw Parquet files and historical report ZIPs are not committed here. Treat all candidate results as development diagnostics, not verified live edge.

| Stage | Result |
|---|---|
| Data acquisition | 37 instruments with data, 75,779,089 reported minute candles; 9 instruments with unresolved repair errors; listing/delisting and TradFi calendars unverified |
| BTC data | 3,628,800 reported minute candles, zero missing minutes; January 2024 native aggregation comparison 3,937 rows, zero reported mismatches |
| Local test suite | 18 passed; 2 grouped as-of sortedness warnings; latest restored code not yet locally retested |
| Compute benchmark | 1/2/4 workers: 2.34/1.81/1.73 seconds; peak tree RSS 138/387/656 MiB; local aggregation only |
| Initial predictive baseline | All 10 model/horizon experiments had negative skill versus naive MSE/Brier benchmarks |
| 60m delay economic screen | 5/44 technical and 2/44 technical+derivatives configurations positive under primary cost assumption; no candidate with both nominal bootstrap lower bounds positive |
| Paired delay sensitivity | 1/5/15/60 minute delay: 9/11/12/7 of 88 positive; best expectancy 0.5679/0.5778/0.7175/0.1605 percent; fixed original predictions and signal schedule |
| 15m retraining | Logistic 4h technical candidate at symmetric 0.70 confidence: 104 trades, net expectancy 0.53817%, PF 2.07294, fixed-notional P&L 55.9697%, realized fixed-notional drawdown 9.3569% |
| Ablation | 12 single-feature removals completed. Without body: 104 trades, expectancy 0.55119%, PF 2.32961, fixed-notional P&L 57.3242%. Without relative_volume_20: 95 trades, expectancy 0.58737%, P&L 55.8003%. Removal of ret_1 reproduced baseline reported metrics exactly; inspect redundancy and prediction identity. |
| Missingness audit | 35,064/35,064 hourly rows; 35,063 complete core rows; 27,066 complete all-feature rows. Top trader ratios ~22% missing, taker ratio ~9%; extended contiguous outages. |
| Current stage | Reduced derivatives completed: 449 trades, expectancy -0.0142576%, PF 0.975505, fixed-notional P&L -6.40168%; both nominal bootstrap intervals cross zero. Technical without body remains exploratory lead. |

Primary cost: 5 bps fee + 1 bp slippage per side, plus funding; funding schedule 4,383 events, 24 missing settlement marks; impacted trades excluded after original selection. Mark minute-open is a proxy. Fixed-notional P&L is NOT compounded portfolio return. Drawdown excludes intratrade floating losses. Bootstrap intervals use 7/30-day calendar blocks, 10,000 replicates; no multiple-testing adjustment yet. The candidate was selected from reused development data. Holdout 2025 reserved, not certified unseen due to prior research exposure. No production readiness claim.

## Artifact registry and reproducibility

Reports under data/reports: 01_data_quality.zip, 02_feature_label.zip, 03_predictive_benchmark.zip, 04_signal_diagnostics.zip, 05_robustness.zip, 06_execution_delay.zip, 07_missing_data.zip. Compute benchmark: data/metadata/compute_benchmark.json. Local archives with browser suffixes (1), (2), etc. are snapshots; keep them outside source control unless explicitly curating a small anonymized research artifact. Code restored to GitHub: baseline ablation with common eligibility, delay_labels.py, delay_engine.py (now excludes already-relabelled experiments), missing_data_engine.py. Source digest changes may produce new experiment IDs; do not automatically retrain or delete previous checkpoints. Local patches may conflict; reconcile using git diff, preserve data and experiment folders, never force reset.

## Next command — reduced derivatives BTC (development diagnostic only)

Windows CMD, repository root with .venv activated. After local/remote reconciliation and pytest:
```cmd
python baseline_engine.py --symbols BTCUSDT --tf 1h --horizon 4 --start 2021-01-01 --holdout-start 2025-01-01 --holdout-end 2026-01-01 --train-days 180 --test-days 30 --threads 2 --seed 42 --model logistic --delay-minutes 15 --aux --drop-features body count_toptrader_long_short_ratio sum_toptrader_long_short_ratio sum_taker_long_short_vol_ratio --export-predictions
python robustness_engine.py
```
Send data/reports/05_robustness.zip. This is a 20-feature comparison using existing train-fold median imputation; not a test isolating imputation. Do not rerun completed experiments solely to change the code hash.

## Remaining gates (supersede older proposed tuning order)

1. Integrity/reproducibility: verify local/remote code, regression tests for ablation mask and alternate labels; partition coverage, finite core features, full timestamp/label continuity, grouped as-of order, feature availability and no leakage. Validate prediction export/checkpoint consistency; preserve prior artifacts. Audit funding boundary/sign and raw-price execution, distinguish missing from stale, no synthetic candles or labels. Recheck revised-delay predictive scores from individual predictions where possible.
2. Missingness experiments: compare technical control and reduced derivatives on identical samples plus full operational coverage. Median as comparator, complete-case, bounded causal carry/rolling median for appropriate state features, native missing handling where supported, technical fallback; use explicit missing flags and all-empty-column policy. No blanket imputations; no stale filling of multi-month outages. Reconstruct taker ratio only if definitions match. Document publication lag sensitivity. Select using development folds, not holdout.
3. Economic validation: complete trade ledger and mark-to-market portfolio accounting, fee schedule/latency/size stress, naive and rule-based trading comparators, annual/regime and conditional direction/calibration diagnostics. Keep fixed notional screening distinct from portfolio return.
4. Feature and target research: fold-local redundant-feature selection, normalized OI/funding/basis/flow, causal regime and multi-timeframe context; return, directional and cost-aware/barrier targets, timeout and same-minute ambiguity; MAE/MFE and exit study. Test hypotheses from uploaded papers rather than assume performance transfers.
5. Cross-asset: repair incomplete instruments, verify listing/calendar and delisted cohorts; shared chronological splits; relative returns, cross-sectional ranking/ProSP and portfolio exposure controls. Evaluate pairs, rule-based technical and macro/news/microstructure branches according to source evidence and data availability. BTC is infrastructure pilot, not a requirement that all research be single-asset.
6. Nested selection and robustness: bounded train-window/threshold/regularization/Optuna studies; parameter plateau only with meaningful neighborhoods, validate on next untouched development slice. Use suitable multiple-testing correction with aligned return series and logged search universe; report negative results. Do not tune just to improve Brier/MSE or maximize reused development P&L.
7. Freeze candidate, use genuinely untouched evaluation or explicitly disclose historical exposure; final cost-aware OOS and forward paper trading. No automatic execution or guaranteed profitability.

## Paper-based research handover

Review full uploaded papers and exact methods before implementing any new experiment. Maintain a paper registry with DOI, market/exchange, sample period, frequency, features, target, validation, execution/cost assumptions, findings, limitations, replication plan and accept/drop decision. Distinguish original findings from adaptations. Earlier reading includes Lin et al. parameter plateau, Jaquart et al. forecasting/trading, Arain & Snudden conditional directional accuracy, Zhang et al. ProSP, Deprez & Frommel technical rules, Cakici et al. cross-sectional ML, Easley et al. microstructure, Albers et al. latency, and crypto pairs/cascade papers. The exact accepted paper inventory must be reconciled with the uploaded full texts; do not assume the list is exhaustive or that papers demonstrate a current Binance futures edge.

## Next checkpoint — controlled common complete-case
Reduced derivatives experiment 2c393ded: 454 original trades, 5 excluded for unknown funding. Compare technical without body and reduced derivatives on identical finite reduced-feature eligibility in each chronological fold. Existing full-coverage reports remain comparators; complete-case selection changes operational coverage and does not isolate imputation alone. No holdout or tuning. New baseline source digest creates new identities; retain previous checkpoints. Implementation commit 057c370; local pytest and runtime acceptance pending.

```cmd
python baseline_engine.py --symbols BTCUSDT --tf 1h --horizon 4 --start 2021-01-01 --holdout-start 2025-01-01 --holdout-end 2026-01-01 --train-days 180 --test-days 30 --threads 2 --seed 42 --model logistic --delay-minutes 15 --drop-features body --missing-policy common-complete --export-predictions
python baseline_engine.py --symbols BTCUSDT --tf 1h --horizon 4 --start 2021-01-01 --holdout-start 2025-01-01 --holdout-end 2026-01-01 --train-days 180 --test-days 30 --threads 2 --seed 42 --model logistic --delay-minutes 15 --aux --drop-features body count_toptrader_long_short_ratio sum_toptrader_long_short_ratio sum_taker_long_short_vol_ratio --missing-policy common-complete --export-predictions
python robustness_engine.py
```
