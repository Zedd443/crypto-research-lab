# Active signal-first research roadmap — 27 November 2026

**This section supersedes all historical next-step commands below.** User clarified that the 100-coin example was intended to refute low trade frequency as a disadvantage of BTC, not request capital allocation research now. Primary objective: discover and validate net signal edge across Binance USD-M assets. No shared wallet, maximum positions, sizing, smart allocation or leverage optimization during signal discovery.

## Current decisions

- C01/C02 remain exploratory controls; integrity/reproducibility passed in prior reports. Report 47 fixed-.70 temperature calibration parked; not evidence against all calibration methods. C03 Channel100 remains a non-ML comparator. No independent holdout or proven live edge.
- Do not eliminate a signal family because trades per coin per month are low. Evaluate per-signal economics, consistency, uncertainty and cross-asset generalization. Record frequency, not a frequency quota. Cross-asset simultaneous opportunities are all retained.
- Separate overlapping signal observations from a within-symbol nonoverlap diagnostic to avoid overstating independent evidence. Do not constrain one coin because another coin has an open position.
- Consolidated entry point and existing historical files remain intact for hashes/checkpoints. No new Python file for each parameter, no report-per-experiment handoff. Only targeted data recovery as justified; no bulk repair.

## Milestones

| Milestone | Deliverable | Status |
|---|---|---|
| M0 | Existing data infrastructure, evidence retention and pilot integrity | Sufficient for existing pilot; broader coverage incomplete |
| M1 | Joint signal-edge search: timeframe, target horizon, exit, threshold and regularization | Implemented; runtime/tests pending |
| M2 | Generalization across point-in-time eligible assets, remaining evidence-backed ML/non-ML families, parameter stability and multiple-testing controls | Pending |
| M3 | Freeze signal protocol; genuinely unseen evaluation or disclose prior exposure; forward signal validation | Pending |
| M4 | Separate risk management/smart allocation research using saved validated signals; shared portfolio, sizing, exposure, margin, fees, order execution and paper trading | Deferred |
| M5 | Live-readiness decision after independent validation and execution/risk gates | Pending; no live orders |

## M1 active batch

Default BTC/ETH/BNB/TRX/NEAR pilot; optional --symbols accepts known crypto subsets. Current full development raw grid and verified 8h funding schedule required; dynamic listing-aware universe expansion remains M2. No automatic downloads. Use frozen binary predictions where available and fit fixed baseline only where no historical comparator exists. Initial bounded search: 30m/1h/4h features × 1/4/8/24h prediction horizon × binary/cost-aware logistic; C=1 coarse then .1/10 on best inner family; thresholds .55/.60/.65/.70 × fixed hold or half-horizon invalidation (exit next minute open). 15m entry delay, 180d rolling train, 120d inner fit and 60d inner validation, 30d outer test; purge by full actual target label end. Same predictions reused for threshold/exit variants. Search is not exhaustive or global optimal; other families/features/delays/windows deferred until evidence supports testing.

Inner selection evaluates chronological nonoverlap opportunities within a symbol (flat at decision timestamp), positive mean net return under both standard and stress costs. No hard minimum trade count or frequency objective. Rank stress expectancy shrunk toward zero by ten zero-return pseudo-observations to account heuristically for sparse evidence; this is an initial design choice, not a statistical significance test. Abstain when no eligible candidate. If inner selected outcomes have unknown funding, mark configuration unscorable, not economically negative. All qualifying outer signals retained in signals.json with probabilities, configuration, timestamps, prices and net returns. No future outcomes used as entry filters. Standard (5bp fee + 1bp slippage each side) and stress (10bp + 5bp) plus funding. Missing settlement marks leave affected outcomes unscored; disclose coverage and conditional known-only metrics, status partial. Intratrade mark-price audit/replay not required for signal discovery.

Report metrics for every qualifying opportunity and separately per-symbol nonoverlap; per-asset, side, fold, signal-weighted and equal-symbol expectancy; trade count, win rate, PF, fixed-unit return sum, nominal 7/30-day calendar-block bootstrap for pooled nonoverlap outcomes. No independent observations claim, no compounded portfolio return, no multiple-testing adjustment yet. Historical development is repeatedly explored. No holdout access. Future smart allocation can consume saved signal ledger without refitting merely to impose capital constraints.

Sequential coordinator, cached raw-derived features, persistent per-symbol/fold checkpoints, identity and independent cost arithmetic checks. Prior standalone files unchanged. Previous batch wallet implementation superseded; no extra Python file introduced for this scope correction. Changing inner objective changes selection identity and legitimately requires a new search if old batch already ran; base dataset cache and historical experiments remain reusable. Tests updated to assert all 100 simultaneous cross-symbol signals retained and unknown funding disclosed. Local tests/runtime pending.

### Active command (Windows CMD)

```cmd
git pull origin main
python -m pytest -q && python research_pipeline.py batch
```

Upload only data/reports/research_batch.zip after batch (pass/partial/fail). If pytest fails, send terminal error. Repeat command to resume compatible checkpoints. Source changes db56c49, b7c04b8, 0347f66.

---

## Superseded shared-wallet proposal (historical design only)

The prior active section proposed shared-wallet replay, fixed five slots and 20% allocations; user explicitly deferred these constraints until after signal-edge discovery. Git history preserves that implementation and documentation; do not run it as the current milestone. Previous source hashes/checkpoints remain intact.

## Historical roadmap and evidence (inactive commands)

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

## Checkpoint — common complete-case results and integrity gate

Source: user-uploaded 05_robustness(1).zip, 40 experiment entries. Primary cost and threshold unchanged (0.70, fee 5 bps + slippage 1 bp per side, funding).

| Candidate | ID prefix | Trades | Expectancy % | PF | Fixed-notional P&L % | Realized DD % |
|---|---|---:|---:|---:|---:|---:|
| Technical without body | 6823506d | 104 | 0.55119 | 2.32961 | 57.3242 | 6.2794 |
| Reduced derivatives median | 2c393ded | 449 | -0.01426 | 0.97550 | -6.4017 | 36.9020 |
| Technical common complete | bd5df1b4 | 103 | 0.38526 | 1.71359 | 39.6817 | 9.0585 |
| Reduced derivatives common complete | e943ad3f | 387 | -0.03060 | 0.94637 | -11.8420 | 39.0314 |

Both technical variants have positive nominal 7/30-day bootstrap lower bounds; both derivatives variants cross zero. Common eligibility does not isolate imputation alone; no conclusion that all derivatives features lack value. Defer further imputation variants until integrity checks and publication-lag/feature-representation audit. No tuning or holdout. Technical remains exploratory lead. Common-complete runtime produced reports; pytest status not supplied.

Next: integrity_engine.py (commit 727e052), read-only with respect to raw data and experiments, sequential monthly raw checks and fold checks, no retraining. Checks current data digest, manifest and checkpoint identities, exact prediction/label keys, raw minute continuity and entry/exit prices, Brier and naive baseline, common train/test masks and economic recomputation. It does not independently prove feature causality, historical publication times, model replay or execution realism; funding ledger helper writes diagnostic report. Local runtime acceptance pending.

```cmd
git pull origin main
python -m pytest -q
python integrity_engine.py
```
Upload data/reports/08_integrity.zip even on audit failure. Preserve all prior checkpoints. Do not rerun completed robustness screening merely to update code hashes.

## Checkpoint — integrity audit PASS (2026-11-26)
Source: 08_integrity.zip. Dataset SHA256 06e9533c3b29d33a567e4f40d9164188c7adf705cc366a9a075e166b0fb14600; 70,118 raw endpoint checks, four experiments verified. Export counts 30,071 / 30,071 / 29,120 / 29,120 (A/B/C/D). Brier 0.25019464 / 0.25537608 / 0.25015834 / 0.25465313 versus naive 0.24994112 / 0.24994112 / 0.24993589 / 0.24993589; all negative skill. DA 52.21 / 51.30 / 52.34 / 51.39 percent. Economic metrics reproduced: expectancy 0.5511946 / -0.0142576 / 0.3852590 / -0.0305995 percent; funding exclusions 0/5/0/0. Preserve detailed results in roadmap and local reports; user requests concise next-step commands only unless detailed analysis requested. Audit is not certification of feature causality or model replay.

Next: feature_causality_engine.py (bebbc22), monthly development-only cached feature replay, sampled prefix invariance and additional auxiliary lag scenarios 0/5/15/60 minutes, with original tolerances. Read-only caches; no model fitting. This tests feature availability under assumptions and coverage/value changes, not actual historical publication timestamps or economic sensitivity. Runtime acceptance pending.

```cmd
git pull origin main
python feature_causality_engine.py
```
Upload data/reports/09_feature_causality.zip even if fail.

## Checkpoint — feature causality PASS
Source: 09_feature_causality.zip; 48 monthly audits passed, cached technical and auxiliary replay and sampled prefix checks matched. Additional assumed lag scenarios 0/5/15/60 minutes evaluated for coverage/value changes only. Historical publication times and economic lag sensitivity remain unverified.

Next: bounded-state imputation (21949d6), same reduced derivatives full eligibility, carry only observed OI, OI value and global long-short ratio for at most one decision hour; no recursive carry; original training-selected missing indicators retained; residual missing values use fold median. This is a hybrid sensitivity experiment, not recovery of missing observations; actual source age can exceed one hour by the original observation lag. No holdout or tuning. Local tests and runtime pending. Economic lag sensitivity remains separate.

```cmd
git pull origin main
python -m pytest -q
python baseline_engine.py --symbols BTCUSDT --tf 1h --horizon 4 --start 2021-01-01 --holdout-start 2025-01-01 --holdout-end 2026-01-01 --train-days 180 --test-days 30 --threads 2 --seed 42 --model logistic --delay-minutes 15 --aux --drop-features body count_toptrader_long_short_ratio sum_toptrader_long_short_ratio sum_taker_long_short_vol_ratio --imputation bounded-state --export-predictions
python robustness_engine.py
```
Upload data/reports/05_robustness.zip. Preserve prior experiments.

## Checkpoint — derivatives V1 diagnosis closed
Source: 05_robustness(3).zip. At threshold .70, fee 5/slippage 1 bps per side plus funding: bounded-state 0442bc85, 451 trades, expectancy +0.0008047%, PF 1.001409, fixed-notional P&L +0.362919%, DD 34.9243%; extra lag 5m d8ee2e9d, 512 trades, expectancy -0.0390005%, PF .927196, P&L -19.9683%; 15m 6e80ff13, 405 trades, expectancy -0.0182728%, PF .966689, P&L -7.40049%; 60m 328f3cf1, 431 trades, expectancy -0.0380415%, PF .931807, P&L -16.3959%. All nominal 7/30 day bootstrap intervals cross zero. Unknown funding exclusions 5/6/4/4 respectively. Park raw derivatives V1; no more imputation/lag tuning on this configuration. Reopen normalized derivatives only under distinct hypothesis. Lag exporter and baseline commits 1fe30bf, ce86a2a.

Next: portfolio_engine.py (da65ba3), incumbent technical 6823506d, 104 trades, threshold .70, primary fixed-notional P&L reconciliation, minute-open mark-to-market equity and daily Sharpe, fixed equity fractions 25/50/100% (sensitivity, not sizing optimization), three fee/slippage scenarios, cash benchmark. Fail on missing intratrade marks or funding; preserve all caches. Funding settlement mark is proxy; no liquidation, margin tiers, intra-minute extremes, or historical fee schedule. Runtime/test acceptance pending. No new model fit or holdout.

```cmd
git pull origin main
python -m pytest -q
python portfolio_engine.py
```
Upload data/reports/10_portfolio.zip even if fail.

Research decision log alignment: incumbent is a benchmark, not exclusive winner. After economic accounting and conditional DA/calibration diagnostics, open controlled candidate branches: technical rules, feature V2, alternative targets/triple barrier, multi-asset and cross-sectional/ProSP, other evidence-driven hypotheses. Read relevant full-text papers before each substantive branch. Only screen-positive candidates enter bounded nested tuning; record search universe and stop rules before runs. Do not use holdout for selection. Separate active/parked/completed/failed branches. No required return target or fixed number of tuning steps. User requests compact execution instructions; retain detailed findings in roadmap.

## Checkpoint — portfolio replay PASS
Source: 10_portfolio(1).zip. 104 trades, 24,960/24,960 valid intratrade mark minutes; zero affected trades after isolated mark recovery. Previous local tests: 18 passed, 2 grouped-asof warnings. Repair commits 05fde7c/e63de7a; diagnostics 3eb2119. Original raw caches untouched. Primary fee/slip 5/1 bps: fixed-notional P&L 57.32424%; compounded return at 25/50/100% notional fractions 15.26845/32.54864/74.03352%; minute-open MTM DD 3.39362/6.65127/12.79002%; annualized daily Sharpe 1.60376/1.60223/1.59919. Exposure 1.37566%. Cost stress 5/3: compounded 14.07287/29.81686/66.95142%; 10/5: 9.99011/20.70013/44.36823%, DD 3.52330/6.90327/13.26601%. Not sizing recommendations or independently validated returns; minute-open marks omit intraminute extremes; funding boundary and cost assumptions remain. Repair provenance metadata local, not included in portfolio ZIP.

Next: conditional_da_engine.py (2b1d80a) frozen incumbent exported predictions only, direction accuracy by confidence/side, reliability bins, ECE, Brier versus naive, per-signal cost-aware returns (overlapping, not portfolio) and selected nonoverlap ledger. Fixed bins are descriptive, not optimized thresholds. No training, holdout or calibration fit. This is a generic diagnostic, not a replication of Arain & Snudden's formal conditional DA methodology; review full paper before any formal replication. Runtime pending. After this diagnostic open candidate expansion, not indefinite incumbent tuning.

```cmd
git pull origin main
python conditional_da_engine.py
```
Upload data/reports/11_conditional_da.zip.

## Checkpoint — candidate expansion: technical rules V1
Conditional DA diagnostics completed (11_conditional_da.zip); higher confidence does not guarantee monotonic economic expectancy. Incumbent 6823506d remains fixed; no threshold optimization. Technical rules V1 (95628e2): 16 fixed configurations (3 SMA, 4 simple rolling RSI, 3 channel breakouts, 6 momentum), plus incumbent and always-long/short controls. Same eligible decision keys, 15m delay, 4h holding, nonoverlap, funding and primary/stressed costs. This is an exploratory adaptation inspired by Deprez & Frömmel, not replication of all six rule classes or their native exit/portfolio selection. No Optuna, no holdout. Report fixed-notional economics and nominal bootstrap; retain all configurations including negative. Advance only plausible positive candidates to native-exit/portfolio evaluation, rather than expand grid automatically. Runtime pending. Conditional DA detailed metrics not yet transcribed here.

```cmd
git pull origin main
python technical_rule_engine.py
```
Upload data/reports/12_technical_rules.zip.

## Checkpoint — fixed-holding technical rules V1 complete
Source: 12_technical_rules.zip; all 16 rules negative after primary and stressed costs. Best per family by expectancy: SMA20/50 -0.107105%, PF .75244 (6027 trades); RSI14/40 -0.106521%, PF .75747 (4448); momentum50/2% -0.103946%, PF .78498 (3575); channel100 -0.056710%, PF .89513 (512). Incumbent reproduced 104 trades, +0.5511946%, PF 2.32961, fixed-notional P&L +57.32424%; stressed +0.370700%, PF 1.75688. Unknown funding exclusions vary by rule (2–20); estimates conditional on known funding. Always-long and always-short controls negative. Reject fixed 4h configurations, not entire rule families. Next: one bounded stateful exit sensitivity on same 16 configurations, no parameter expansion. Channel holds until opposite breakout; SMA/momentum/RSI use signal state, flat on neutral, reverse on opposite; exit delay 15m, terminal close on final eligible decision. Same common decision grid, gaps may delay exits (reported); no claims of fully native continuous-market execution or exact paper replication. Fee/slip 0/0 (funding remains), 5/1 and 10/5. Report duration/exposure and missing funding exclusions. If positive candidates appear, verify funding coverage and continuous hourly execution and MTM before tuning; otherwise park these rule implementations and move to feature/target research, retaining OBV/filter/other paper-specific hypotheses as untested. Source: Deprez & Frömmel 2024 appendix B, adaptations not replication. Runtime pending; no holdout or retraining.

```cmd
git pull origin main
python stateful_rule_engine.py
```
Upload data/reports/13_stateful_rules.zip. Script commit bd7e043.

## Checkpoint — stateful technical screening
Source: 13_stateful_rules.zip, prior report assessment: Channel100 123 included trades, expectancy +1.1085%, PF 1.423, fixed-notional P&L +136.34%; stressed P&L +114.03%; four funding-affected trades excluded, mean holding approximately 237h. SMA50/200 and Channel50 positive at primary costs but not stressed costs. Park remaining 15 configurations; retain Channel100 as exploratory challenger, not a replacement for Logistic. Exposure and risk differ materially. Next: channel_audit_engine.py (3821862), same first/last incumbent decision dates but independent continuous hourly grid, causal 100-bar channel, 15m execution delay, raw minute-open fills, reversal on opposite breakout. Fail on missing signal or execution coverage; report missing funding events intersecting all selected trades, without dropping trades. If funding incomplete, targeted recovery required before portfolio accounting. No parameter tuning or holdout. Runtime pending; detailed source results should be verified against original report before final selection.

```cmd
git pull origin main
python channel_audit_engine.py
```
Upload data/reports/14_channel_audit.zip regardless of status.

## Checkpoint — Channel100 independent execution and funding PASS
Source 14_channel_audit(1).zip: 127 trades, no missing hourly decisions, warmup or execution endpoints; all 24 missing BTC funding marks recovered (21 affected four original trades), no excluded trades. Recovery commits 8aa1f2c/9f3a714. Exposure 99.6089% on channel audit window. Primary 5/1 bps: expectancy 1.324837%, PF 1.508962, fixed-notional P&L 168.25434%, DD 40.50147%, WR 37.7953%; annual fixed-notional P&L 2021 +46.44557%, 2022 -8.47173%, 2023 +76.10405%, 2024 +54.17645%. Nominal block CI 7d [-0.38193%,3.25076%], 30d [-0.30123%,3.28184%]. Stressed 10/5: P&L +145.19776%, expectancy 1.143289%, PF 1.420998, DD 42.49349%; both intervals cross zero. Source hash 96128d59834b0d4b8e336f121b69706c58adaad086a566785605cd142c0e40e8. Funding marks remain minute-open proxies. Channel is challenger, not validated winner; Logistic has much lower exposure. Next channel_portfolio_engine.py (3e744ad): audited ledger, monthly bounded scans, common incumbent manifest calendar, primary/stress costs and 25/50/100% fixed notional fractions, complete minute-mark coverage required. Report gaps without dropping trades, no liquidation model or intraminute extrema, no sizing tuning. Runtime pending. No holdout.

```cmd
git pull origin main
python channel_portfolio_engine.py
```
Upload data/reports/15_channel_portfolio.zip regardless of status. After portfolio accounting, park or retain challenger and proceed to Feature V2 rather than tune immediately.

## Checkpoint — Channel100 portfolio PASS
15_channel_portfolio(3).zip: 1,807,020/1,807,020 active mark minutes valid, zero gaps. Recovery commits 73d36c1,52e15fb,421768b. Primary 5/1 bps, fractions .25/.5/1: compounded returns 46.29554/98.76225/202.73316%, MTM DD 12.76383/23.83470/41.97567%, Sharpe .88217/.88443/.88330. Stress 10/5: returns 38.12796/77.23118/140.76425%, DD 13.54397/25.19785/44.05701%, Sharpe .75556/.75578/.75135. Exposure 99.59325% on manifest calendar. All scenarios pass; no liquidation model or intraminute extremes. Logistic retains incumbent status on risk-adjusted performance; retain Channel100 as exploratory non-ML challenger, no tuning. Close technical V1, open Feature V2 with literature-based hypotheses and bounded resource usage. No holdout.

## Feature V2 — normalized derivatives pilot
Script feature_v2_engine.py (eb5725a). BTC hourly 4h horizon 15m delay, frozen incumbent folds, hyperparameters, seed, eligibility and .70/.30 threshold. Add five causal representations: log OI changes 1h/24h, funding rate minus prior 168 hourly mean, premium minus prior 24 hourly mean, log global long-short ratio change 1h. OI/ratio exact assumed 5m age, gap-aware shifts; training-only median/indicators and scaling. Hourly funding mean weights repeated as-of rates, not independent settlement events. Compare cached incumbent predictions and new challenger; same test keys and targets asserted; recovered funding ledger, primary and stress costs. Preserve original feature/label caches, separate identity-keyed features and resumable fold predictions. This is a predeclared exploratory engineering hypothesis, not a verified replication of a specific paper; full-text literature mapping remains pending. Runtime not verified. No tuning or holdout. If negative, retain evidence and advance to next feature group rather than optimize this group indefinitely.

```cmd
git pull origin main
python feature_v2_engine.py
```
Upload data/reports/16_feature_v2.zip regardless of status.

## Checkpoint — normalized derivatives V2 PASS, park
Source 16_feature_v2.zip, experiment 151050258aa17feb380dc6ef7cc0b66bbe214b851f5c1d1abca788f4113a528f. Common eligibility verified. 35,064 feature rows; missing counts 113/172/168/355/540 for five features respectively. Challenger 268 trades, expectancy +0.061419%, PF 1.109586, fixed-notional P&L +16.460328%, DD 43.214029%, stress P&L -31.811874%, DA 52.3461%, Brier .25537998 vs incumbent .25019464 and naive .24994112. Nominal 7d and 30d CI cross zero; incumbent unchanged. Park normalized derivatives V2 (not all possible derivatives hypotheses). Next market_regime_engine.py (b69cf57): 24h/168h realized hourly volatility ratio, signed directional efficiency (sum log returns / sum absolute log returns) at 24h/168h, close location in trailing high-low range 24h/168h. Fixed windows, causal completed hourly bars, gap-aware segments, same incumbent features/folds/threshold, no tuning. Overlap with existing trend features possible; test incremental contribution. Separate identity-keyed cache, no holdout. Exploratory engineering adaptation, not exact paper replication; full-text literature mapping pending. Runtime pending.

```cmd
git pull origin main
python market_regime_engine.py
```
Upload data/reports/17_market_regime.zip.

## Checkpoint — market regime V2 PASS, secondary candidate
Source 17_market_regime.zip, experiment 3c6aaa8583ead14b4921940585a01b34f475beb8395a25499f5ed07e54f751f1. Common OOS eligibility verified by engine. 35,064 feature rows, missing 168/24/168/23/167. Market regime: DA 53.4601%, Brier .249942292 vs naive .249941121; 203 trades, expectancy +.2413906%, PF 1.487961, fixed-notional P&L +49.002282%, DD 9.058472%, WR 54.1872%. Nominal 7d CI [.020137%,.471298%], 30d [.019219%,.486612%]. Stress: expectancy +.0611737%, P&L +12.418258%, PF 1.104423, DD 12.096697%; both CIs cross zero. Incumbent superior economically. Retain secondary candidate for later controlled comparison, no tuning. Next liquidity_engine.py (5282ca3): five predeclared liquidity proxies: ratio of 24h/168h mean absolute log return per quote volume (Amihud-style adaptation), quote volume and trade count ratios of 24h/168h means, signed taker quote flow / total quote volume over 4h and 24h. Same incumbent folds, labels, base features, threshold, costs, seed and hyperparameters; gap-aware completed OHLCV; no order-book or actual market impact claim. Some redundancy with base flow/volume features is possible; test incremental contribution. Literature mapping pending, exploratory not exact replication. Runtime pending; no holdout or tuning.

```cmd
git pull origin main
python liquidity_engine.py
```
Upload data/reports/18_liquidity.zip.

## Checkpoint — liquidity V2 PASS, park
Source 18_liquidity.zip, experiment db2321fc13f6ec7039b34fe8861d4df8464c4ba133129e3b55cf153ef968d2e0. 35,064 rows; missing 336/167/167/3/23. DA 52.2630%, Brier .250589237 vs incumbent .250194639 and naive .249941121. Primary: 144 trades, expectancy +.2223266%, PF 1.480846, fixed-notional P&L +32.015026%, DD 7.208729%, WR 47.9167%; stressed P&L +6.043164%, PF 1.074851, DD 12.441763%. All nominal bootstrap intervals cross zero. Park liquidity V2, retain market regime as secondary candidate. Next market_structure_engine.py (e721515): five fixed additions, log close distance from prior 20/100 hourly highs and lows, log channel width ratio 20/100. Exclude current candle from channel bounds, gap-aware segments; same incumbent folds, eligibility, label, delay, threshold and costs. Inspired by earlier channel breakout evidence, not exact paper replication; literature mapping remains pending. No tuning or holdout; runtime pending. Next multi-timeframe features.

```cmd
git pull origin main
python market_structure_engine.py
```
Upload data/reports/19_market_structure.zip.

## Checkpoint — market structure V2 PASS, park
Source 19_market_structure.zip, experiment 10ccd7ac93835947a0e3b172f67c99933be4ad58b70cb898a46d84a2a9931221. 35,064 rows, missing 20/20/100/100/100. DA 52.415949%, Brier .250750311. Primary: 181 trades, expectancy .1990086%, PF 1.351700, fixed-notional P&L 36.020551%, DD 13.134184%; stress P&L 3.390908%, PF 1.028837, DD 17.564786%. All nominal bootstrap intervals cross zero. Park market structure V2. Next multi_timeframe_engine.py (1793279), six frozen additions: completed 4h return over 6 bars, SMA20 gap and volatility20; completed daily return over 7 bars, SMA20 gap and volatility20. UTC-aligned aggregation from complete hourly bars, gap-aware rolling, causal backward join restricted to age less than one higher-timeframe bar. No incomplete or future bars. Same incumbent folds, eligibility, labels, threshold and costs. Exploratory engineering hypothesis, not verified journal replication; full literature mapping pending. Runtime unverified. No tuning or holdout. Close initial feature V2 screening after this run; then target engineering.

```cmd
git pull origin main
python multi_timeframe_engine.py
```
Upload data/reports/20_multi_timeframe.zip.

## Checkpoint — multi-timeframe V2 PASS, close initial Feature V2
Source 20_multi_timeframe.zip, experiment 1ce4dd71966e1f695082d062572f53d9c13fe2da5c1819791215beaf26593ae1. 35,064 rows, missing 27/79/83/191/479/503; DA 52.110006%, Brier .250852772 vs naive .249941121. Primary 171 trades, expectancy +.149939%, PF 1.254956, P&L +25.639561%, DD 15.465544%; stress P&L -5.169351%, PF .955528, DD 22.440898%. All nominal bootstrap intervals cross zero. Park multi-timeframe. No V2 group beat incumbent; retain regime as secondary candidate only. Next cost_aware_target_engine.py (8d35da1): fixed three-class logistic, labels short/flat/long determined by positive realized hypothetical net return at 5/1 bps plus recovered funding; flat if neither profitable. Same incumbent feature set, fold purge, 4h horizon, 15m delay, seed/C, threshold .70 for either profitable side (not equivalent signal coverage to binary .70), no overlap. Evaluate primary/stress economics and multiclass Brier vs train class priors, not binary Brier directly. Funding only in labels, never predictor; cache keyed by data, code, funding and libraries. No holdout/tuning. Exploratory target adaptation, full-text mapping pending; runtime pending. Next triple-barrier design with gap-aware minute paths and explicit ambiguity rules.

```cmd
git pull origin main
python cost_aware_target_engine.py
```
Upload data/reports/21_cost_aware_target.zip.

## Checkpoint — cost-aware target PASS, retain
Source 21_cost_aware_target.zip: 63 trades, expectancy +0.6908%, PF 2.802, fixed-notional P&L +43.52%, DD 4.75%, WR 60.32%; stress expectancy +0.5099%. Nominal bootstrap intervals positive under both cost scenarios. Multiclass Brier marginally better than class-prior baseline. Retain alongside binary incumbent, not a replacement. Next triple_barrier_engine.py (e253532): symmetric log-price barriers max(0.0012, 2 times past hourly volatility20), checked on completed minute closes, exit next minute open or at 4h maximum. This is a closed-bar adaptation rather than intrabar TP/SL execution; no assumed fill at barrier price or ambiguous intrabar ordering. Cost-aware classes based on realized net return at actual exit. Preserve fixed-horizon fold eligibility and conservative purge (actual exit also checked), same feature set and threshold .70; compare original incumbent, exit-only incumbent with barriers, and retrained three-class target. Validate raw minute path continuity and original endpoints, cache labels and predictions, no tuning or holdout. Exploratory parameters chosen in advance, not optimized; literature replication and runtime unverified.

```cmd
git pull origin main
python triple_barrier_engine.py
``
Upload data/reports/22_triple_barrier.zip regardless of status.

## Checkpoint — triple barrier PASS, park
Source 22_triple_barrier.zip, experiment dae053159060075148792924c01a8dd4eec71ab0480514037663c0933479d9ce. OOS class counts short/flat/long 12899/3928/13244. Multiclass Brier .603344225 vs prior .604393726, accuracy 44.365%. Retrained barrier: 29 trades, expectancy .177006%, PF 1.189877, P&L 5.133185%, DD 14.291142%; stress P&L -.099116%, both nominal CIs cross zero at both costs. Incumbent with barrier: 124 trades, expectancy .387040%, PF 1.622795, P&L 47.992968%, DD 12.171551%; stress P&L 25.614884%, nominal CIs positive at primary but cross zero at stress. Exit-only comparator allows reentry after earlier exits, so not a paired same-trade exit comparison. Park both tested barrier configurations; not a rejection of all adaptive exits. Next excursion_engine.py (0145b4c): read frozen binary and cost-aware predictions, selected 104/63 trades; validate original raw minute coverage and endpoints, compute gross MAE/MFE/giveback and timing quantiles by wins/losses; compare overlapping decision keys. No executable extrema or intrabar order assumptions, no optimized TP/SL; diagnostics only. Requires local 21_cost_aware_target.zip and cached predictions. Runtime pending; no holdout or retraining. After this close initial target diagnostics and move to multi-asset/cross-sectional expansion, unless a material integrity issue arises.

```cmd
git pull origin main
python excursion_engine.py
```
Upload data/reports/23_excursions.zip.

## Checkpoint — excursion diagnostics PASS; close initial target branch
Source 23_excursions.zip, source hash 6c9a9ea5a212657ce3211ccee31f1ec443bc519c836e907d18945dd9c6836b13. Frozen 104 incumbent /63 cost-aware trades, median gross MAE .94651/1.00477%, MFE 1.38283/1.48099%, giveback 1.00179/1.03372%. Winners median MAE .64753/.74227%, MFE 2.25694/2.35001%, time to MFE 156.5/143.5 minutes; losers time to MFE 39/29 minutes. 56 overlapping decision timestamps, not proof of same side or independent diversification. No ex-post TP/SL selection. Retain fixed 4h candidates. Next multi_asset_readiness_engine.py (78edf83): sequential monthly raw-minute audit on 30 crypto symbols, development only, full-window conservative readiness criteria and funding/mark partition presence. No synthetic bars, downloads, model fits or holdout. Existing present-day universe is selection-biased; full-window gate excludes later listings and is only for transfer pilot, not cross-sectional universe construction. Listing dates, actual funding completeness and mark coverage still require verification before economic evaluation. Runtime pending. Next test frozen BTC specification on eligible independent symbols, with appropriate label/feature generation and funding audits; cross-sectional ranking separately after literature review.

```cmd
git pull origin main
python multi_asset_readiness_engine.py
```
Upload data/reports/24_multi_asset_readiness.zip.

## Checkpoint — multi-asset preliminary readiness
Source 24_multi_asset_readiness.zip: 10/30 pass full-window raw-minute and partition-presence gate (BTC, ETH, BNB, DOGE, ADA, LINK, AVAX, BCH, UNI, ATOM), each 2,103,840 raw minutes, zero internal gaps/invalid rows, 48 funding and mark partitions. Nine symbols have exactly 7,200 missing minutes (SOL, XRP, TRX, LTC, NEAR, HBAR, WAVES, XEM, OMG); investigate common missing dates later. XRP/DOT/SUI end at October 2023 despite expected later availability; AAVE/FIL absent in development; other partial coverage includes listing effects. Do not infer listing/delisting from cache alone. Next multi_asset_transfer_engine.py (1d7ec80): sequential independent per-symbol retraining of frozen BTC specification on nine passing alternatives (not zero-shot transfer), technical-only hourly features built from raw minute master, exact 15m delayed open entry and 4h open exit, same 42 folds, original C/seed and .70 threshold. Cached datasets and fold predictions, no holdout/tuning/download. Report predictive skill and explicitly preliminary fee/slippage-only diagnostics; funding not assumed zero for economic conclusions. BTC cached predictions are fee/slippage-only control. Funding schedules, settlement marks and listing history must be verified before net profitability claims. Full-window survivor selection bias remains, not a cross-sectional portfolio. Runtime pending.

```cmd
git pull origin main
python multi_asset_transfer_engine.py
```
Upload data/reports/25_multi_asset_transfer.zip.

## Checkpoint — multi-asset transfer PASS
Source 25_multi_asset_transfer.zip: all 9 alternative symbols completed 42 folds with frozen BTC specification. Fee/slippage-only primary expectancy (%): BTC .553738, BNB .612890, ETH .374393, ATOM .253648, AVAX .187527, UNI .123457, ADA -.236281, DOGE -.250194, BCH -.473737, LINK -.545440. Stress fixed-notional P&L (%): BTC 38.8171, BNB 43.6337, ETH 15.1339, ATOM 7.5370, AVAX .5387, UNI -4.5405, ADA -37.9160, DOGE -79.1279, BCH -129.6206, LINK -45.6710. All altcoin nominal bootstrap intervals cross zero at both costs. Only DOGE marginally improves Brier against its naive baseline, despite negative economic performance. BNB primary P&L +61.9019%, fixed-notional DD 30.9787%; 2024 contributes +56.7847 percentage points. BTC retains incumbent; BNB/ETH exploratory candidates, no generalization claim. Negative results retained. Next multi_asset_funding_engine.py (545c2d3): rescore all five primary-positive altcoins with no retraining or parameter changes. Strict 8h funding schedule gate as initial check, report any nonstandard intervals for separate historical verification, do not silently assume 8h or fill missing events. Missing settlement marks affecting trades block net scoring pending targeted repair. Minute-open mark proxy, fixed-notional not compounded portfolio, no holdout. Runtime unverified; listing history remains unverified. Conditional on selected full-window universe and positive-primary screening; no multiple-testing adjustment.

```cmd
git pull origin main
python multi_asset_funding_engine.py
```
Upload data/reports/26_multi_asset_funding.zip regardless of status.

## Checkpoint — multi-asset funding PASS; close independent screening
Source 26_multi_asset_funding.zip: all five pass standard 8h grid, 4383 events each, no missing/duplicate/off-grid settlements. Missing minute-open marks BNB/ETH/ATOM/AVAX/UNI = 24/6/6/24/21; none intersect selected trades. Primary net expectancy (%) .623735/.376365/.261886/.199131/.127296; PF 1.563867/1.461428/1.193856/1.128860/1.105862; fixed-notional P&L (%) 62.99719/29.35650/26.71241/15.53218/10.05640; DD (%) 30.88060/18.61210/36.44311/46.24050/49.01200. Stress P&L (%) 44.72853/15.28773/8.37701/1.44345/-4.23718. All nominal bootstrap intervals cross zero. BTC remains incumbent, BNB/ETH exploratory only, park ATOM/AVAX/UNI. No additional funding repair required for these selected trades, but coverage not sufficient for arbitrary new trades. Next cross_sectional_engine.py (4fbdd85): fixed balanced ten-asset survivor panel from cached independent logistic OOS probabilities; Spearman IC on hourly common eligibility, fixed UTC 4h schedule, equal-weight top/bottom two (gross notional one), compare 24h momentum/reversal controls. No tuning/retraining, no holdout. Fee/slippage-only preliminary basket economics, funding must be audited separately for any viable candidate. Ranking probabilities trained independently are not expected-return forecasts and may not be comparable across assets. Universe selection bias and ex-post balanced coverage limit inference. Exploratory adaptation, not ProSP replication; full-text literature mapping still pending. Runtime pending.

```cmd
git pull origin main
python cross_sectional_engine.py
``
Upload data/reports/27_cross_sectional.zip.

## Checkpoint — cross-sectional V1 PASS, park
Source 27_cross_sectional.zip: 30,071 balanced hourly cross sections, 7,518 scheduled baskets; logistic/momentum/reversal IC .0177675/-.0249181/.0249181, primary expectancy (%) -.1124488/-.1126950/-.1273388, PF .591517/.629519/.586825. Nominal bootstrap intervals entirely negative; stress expectancy (%) -.292482/-.292727/-.307370. All fee/slippage-only. Park tested configurations. Literature: Zhang, Ji, Ma & Cao (2026), Probability-based portfolio in crypto-perpetual futures market, DOI 10.1016/j.physa.2026.131911, local full-text PDF S0378437126006473. Algorithm 1 and Section 5 use three relative-return classes, dual 25% tail probability sorting, remove intersections; monotonicity derived under location-scale assumptions, not unconditional across heterogeneous volatilities. Sections 6.4 and Conclusion report 8h XGB-ProSP annualized net return 44.32% at 1 bp per side, but all tested strategies unprofitable at 5 bps per side; gross results not sufficient for our 5+1 bps assumptions. Adapt rather than replicate: prosp_engine.py (c96b76a), pooled multinomial logistic on existing technical features and same 42 purged folds, fixed ten-asset balanced panel, two assets per realized tail and per candidate leg, dual-sort overlap removal versus single-score p_top-p_bottom comparator. Fixed 4h horizon and schedule, 15m delay; equal gross notional one on surviving baskets, cash on empty baskets. Report gross, 1bp, 5+1 and 10+5 bps per side; funding not yet included. This differs from paper's larger changing universe, feature sets, model architectures and 4h/8h protocols. No tuning/holdout. Runtime unverified. If economically negative, park tested adaptation rather than escalating model complexity automatically. If promising, audit funding and turnover before any model expansion.

```cmd
git pull origin main
python prosp_engine.py
```
Upload data/reports/28_prosp.zip.

## Checkpoint — ProSP V1 PASS; eliminate tested configuration from tuning
Source 28_prosp.zip, experiment ce693e976a4c3c2575ae36cc0b51d44b44cc7a1a50ad05d4b67cb8db6a106642. Dual/single gross expectancy (%) .005360/.000397, at 1bp -.003907/-.019605, at standard 5+1bps -.050243/-.119618; standard PF .666047/.585253. 7518 scheduled baskets, dual 4035 flat, 11237 overlap pairs removed. Brier .551716 vs .56 prior. Both standard-cost nominal CIs negative, no funding. Eliminate this 11-feature configuration from tuning; no complexity escalation. Original feature specification remains a separate bounded test. Rechecked local paper S0378437126006473, Section 3.3 equations 1–6 and Table 3: Set 1 28 dims (current return/turnover, five cumulative return and turnover windows, 12 lagged normalized returns, four normalized cumulative returns); Set 2 12 dims (log basis, five 7–35 day basis momentum, price-volume and five variance windows). Quote volume candidate proxy for turnover; exact vendor definition and equation 6 variance notation require explicit interpretation. Section 6.4 favors 8h but reports cost sensitivity. Next prosp_readiness_engine.py (0253956): audit exact aligned spot closes and futures volume inputs at 4h/8h, conservative 35-day contiguous endpoint history, balanced coverage across ten survivor assets. No training, downloads, or holdout. Runtime pending; spot endpoint availability is not full publication-time verification.

### Candidate registry (exploratory; no final deployment approval)
C01 BTC logistic technical incumbent; C02 BTC logistic cost-aware primary challenger; C03 BTC channel100 secondary non-ML; C04 BTC market regime reserve; C05 BNB logistic and C06 ETH logistic exploratory reserves. C01/C02 priority for future bounded nested tuning only after remaining research and integrity/portfolio gates; C03–C06 require additional evidence. Eliminate previously failed configurations from tuning, preserve negative evidence and raw data; no paper, model family or asset permanently rejected. P1DR 24h horizon and pairs trading remain pending. No holdout-based selection.

```cmd
git pull origin main
python prosp_readiness_engine.py
```
Upload data/reports/29_prosp_readiness.zip.

## Checkpoint — ProSP readiness PASS, partial spot coverage
Source 29_prosp_readiness.zip: ten assets each 8766/4383 complete futures bars at 4h/8h; nine assets have identical five/two missing spot endpoints, leaving conservative 7676/4063 complete-history periods each. ATOM has zero spot partitions, hence ten-asset Set3 coverage zero. Set1 readiness 8753/4370 periods. These are feature coverage counts, not OOS rows. Next prosp_original_features_engine.py (01e9cb9): six bounded runs, 4h/8h each with Set1 ten assets, Set1 nine and Set3 nine matched on spot-ready common sample. Paper equations 1–5 and Table 3, lagged returns/turnover, normalized cross-sectional returns; quote volume proxy for turnover, ordinary sample variance of price times base volume as interpretation of ambiguous Eq6, no synthetic spot. Two tails each, same pooled logistic C/seed and purged folds, delayed open-to-open labels, fee/slippage-only 0/1/6/15 bps per side, no tuning or holdout. Not exact replication; runtime pending. ATOM remains in universe, no spot recovery now. If no promising edge, eliminate tested feature adaptation from tuning and proceed P1DR 24h.

```cmd
git pull origin main
python prosp_original_features_engine.py
```
Upload data/reports/30_prosp_features.zip.

## Checkpoint — ProSP original-feature adaptation PASS; eliminate tested configurations from tuning
Source 30_prosp_features.zip: all six runs pass. In order 4h Set1_10, Set1_9, Set3_9, 8h Set1_10, Set1_9, Set3_9: gross expectancy (%) .02120/.01190/.00091/.01129/.02402/.03907; standard fee/slippage expectancy (%) -.08162/-.09586/-.10745/-.09629/-.08651/-.07237; PF .702/.650/.621/.762/.773/.806. Best 8h Set3_9 gross nominal bootstrap intervals positive but at 1bp per side intervals cross zero. All standard-cost negative before funding. No additional tuning budget for tested ProSP adaptations, not a rejection of the original paper. Next p1dr_daily_engine.py (64f79c2): Liu et al. (2023), DOI 10.1016/j.ribaf.2023.101905, daily lagged return reversal hypothesis, ten survivor futures assets, UTC daily 24h lookback and 24h holding, 15m delay, existing purged development evaluation windows, top/bottom two equal-weight legs, momentum sign control, scheduled IC and gross/standard/stress fee-slippage economics. No training, funding yet, tuning or holdout. Smaller universe and different market from original study; runtime unverified. If negative, eliminate tested daily ranking configuration and proceed pairs trading.

```cmd
git pull origin main
python p1dr_daily_engine.py
```
Upload data/reports/31_p1dr_daily.zip.

## Checkpoint — daily P1DR PASS; eliminate tested ranking configurations
Source 31_p1dr_daily.zip (previous conversation report): 1,218 baskets each; momentum/reversal Spearman IC -/+ .01743, gross expectancy -/+ .03027%, standard fee/slippage expectancy -.15030%/-.08975%, PF .775/.858. Gross reversal CI crosses zero; standard-cost CI negative. Funding not included. Eliminate tested 4h and 24h ranking configurations from tuning, not the original Liu hypothesis. Candidate registry unchanged C01–C06.

## Data integrity clarification and pairs pilot
Repository universe.py defines 37 instruments: 30 crypto plus 7 TradFi. Ten crypto passed strict full-window 2021–2024 readiness, not a complete point-in-time universe. Twenty other crypto include later listings, delistings and incomplete caches; nine unresolved repair errors were recorded in earlier acquisition audit. Do not equate missing historical coverage with actual outages. BTC integrity, funding and mark repairs passed scoped audits, not certification of all datasets. Before final candidate selection, complete listing/delisting-aware coverage and targeted repair audit, verify raw archive provenance and publication assumptions, and re-evaluate universe-dependent conclusions if coverage changes materially. Preserve immutable raw data and existing checkpoints.
Next cointegration_engine.py (1e40c3c): nine BTC-reference pairs, completed hourly futures price levels, training-only Engle–Granger with intercept, ADF levels/differences, Holm correction across nine pairs within each fold, positive hedge ratio and AR1 half-life diagnostic. Existing 42 rolling 180d training windows; no test fitting, no holdout, no backtest. ADF non-rejection is not proof of I(1); within-fold Holm does not correct all research trials. Exploratory adaptation of Tadi & Witzany, DOI 10.1186/s40854-024-00702-7; not exact replication. Runtime unverified. Install statsmodels if required; cached fold results. Next causal spread trading benchmark before copula complexity.

```cmd
git pull origin main
python -m pip install statsmodels
python cointegration_engine.py
```
Upload data/reports/32_cointegration.zip.


## Checkpoint — 27 November 2026: research review and targeted repair
Report 32_cointegration.zip: 378 pair-fold tests; only ETH passes all gates in one of 42 folds (half-life 147.05h); no robust pair candidate, no copula tuning. Report 33_invalid_candles.zip: DOT/SUI/XRP each have inconsistent taker base volume exceeding total at 1701347700000, identical monthly/daily/REST. Exact-row quarantine patch 0ce6325; user terminal confirms 43,199 valid November 2023 minutes each, one quarantined gap, Dec 2023–Dec 2024 complete. Earlier daily fallback patch 3d48b01 recovered 64,800 minutes across nine symbols (Feb/Apr 2022); AAVE/FIL now full development OHLCV, LOOM observed from Oct 2023. Listing dates not independently verified. Auxiliary repair in progress: spot ATOM/HBAR/NEAR/DOT/XRP, premium SUI. Report 24 predates repairs: funding/mark absent for AAVE/FIL and truncated for DOT/XRP/SUI/LOOM; acquire missing auxiliary partitions before full economic evaluation, audit all kinds and actual settlement schedules. Do not rerun known quarantined rows or complete partitions.
Review 01–31: C01/C02 remain primary, C03/C04 secondary, C05/C06 exploratory. C02 needs comparable mark-to-market portfolio audit. Next after auxiliary repair: listing/delisting-aware gap-aware readiness and incremental transfer on SOL/TRX/LTC/NEAR/AAVE/FIL/DOT/XRP, HBAR subject to verified start; later-listed SUI/ORBS/LOOM/WIF/ENA/PNUT/VIRTUAL/PENGU and delisted WAVES/XEM/OMG require point-in-time eligibility, no forced full-window requirement. These are proposed test groups, not confirmed readiness. Preserve prior negative configs. Reopen Liu P1DR hypothesis only for expanded point-in-time universe, liquidity/return-tail diagnostics and bounded P1DR-only OLS/nonlinear comparator; prior ranking remains eliminated from tuning. ProSP adaptations remain parked unless materially new evidence. No holdout access, no tuning until integrity and portfolio gates; multiple-testing adjustment and independent forward evidence outstanding.

## Checkpoint — auxiliary retrieval and post-repair audit
User terminal: spot ATOM/HBAR/NEAR/DOT/XRP all cached (48/46/48/48/48 months), not proof of observations; SUI premium Aug 2023–Dec 2024 retrieved, November 2023 short 14 minutes and August 2024 short 2 relative to full month. OHLCV audit completed for 37 instruments. Next post_repair_coverage_engine.py (c53ab5f), sequential monthly local audit of 30 crypto in development only, raw validity and observed bounds, auxiliary actual grid/duplicate/nonfinite counts and funding observed within-month intervals; report review queue, not automatic download queue. Listing/delisting, funding schedule, cross-month funding intervals, publication time and auxiliary OHLC consistency not certified. Runtime unverified. BTC remains default pilot; cross-sectional hypotheses require multi-asset universe. No retraining or holdout.

```cmd
git pull origin main
python post_repair_coverage_engine.py
```
Upload data/reports/34_post_repair_coverage.zip.


## Checkpoint — Report 34 PASS; prioritize missing economic inputs
34_post_repair_coverage.zip: 8202 partition records, 2021 review items, zero runtime errors. 18/30 crypto have 48 full raw months: original ten plus SOL/TRX/LTC/NEAR/AAVE/FIL/WAVES/OMG. DOT/XRP each 47 full and one partial; listing/delisting remains unverified. Auxiliary absent mark and funding each: AAVE/FIL 48 months each, DOT/XRP/SUI 14 each, LOOM 9. Metrics 448 INVALID (null/nonfinite or duplicates, not necessarily corrupt whole partition), 357 absent, 72 partial, 250 complete grid. Funding 980 observed partitions, schedule unverified. ATOM spot now present, partial gaps remain. Prioritize economic inputs only; defer other auxiliary gaps until needed by hypothesis. Next targeted acquisition using existing derivatives_engine.py with no forced repair on complete partitions. AAVE/FIL 2021-01 to 2025-01; DOT/XRP/SUI 2023-11 to 2025-01; LOOM missing months to be checked separately before acquisition. Cached zero-row markers may need targeted --repair; verify actual files with report 34 rerun. No assumption of standard funding intervals, no tuning/holdout. BTC remains pilot.


## Checkpoint — stop general repair; consolidate entry points; C02 portfolio
User directed consolidation and return to strategy research. Supersede unconditional auxiliary acquisition commands above: no need to execute them now. Repair on demand according to hypothesis and selected economic inputs; preserve all raw data and negative evidence. BTC remains default pilot. Existing research_pipeline.py now exposes coverage, invalid-candles, repair-funding, repair-marks, repair-channel-marks and candidate-portfolio (96ef255). Historical modules remain physically intact to preserve imports, source hashes and cached experiments; CLI consolidation is not physical deletion. portfolio_engine.py (f5a70d6) adds --candidate compare, reusing mark audit and replay for frozen C01/C02, common original fold calendar, 104/63 trades, three fee/slippage scenarios 5/1,5/3,10/5 and fixed notional fractions .25/.5/1. Verify common eligibility, prediction scope, probability validity, funding hash, trade counts, nominal P&L reconciliation and mark coverage before replay. Existing report 10 default unchanged. No retraining, downloads, holdout or sizing optimization. Missing mark coverage blocks replay and yields targeted issues. Runtime unverified; minute-open proxies, no liquidation/margin tiers, no multiple-testing adjustment. Next evaluate report 35, repair only affected mark minutes if needed, then continue bounded BTC exit/technical hypothesis screening and incremental eligible universe transfer.

```cmd
git pull origin main
python research_pipeline.py candidate-portfolio
```
Upload data/reports/35_candidate_portfolio.zip even if failed.


## Checkpoint — Report 35 PASS; paired exit diagnostic
35_candidate_portfolio.zip verified: C01/C02 104/63 trades, mark coverage 24960/15120 of expected minutes, zero affected trades. At fraction 1, primary compounded return 74.0335%/52.4671%, MTM DD both 12.7900%, daily annualized Sharpe 1.59919/1.42300, exposure 1.37566%/.83333%; stress return 44.3682%/36.1312%, DD both 13.2660%. C01 retains incumbent, C02 challenger, no deployment claim. Next Report 36: paired fixed-entry time exit 120/180/240 minutes, same frozen trades and directions from report 35, no new entries after earlier exit, exact minute open exit, funding rescored, common calendar, 5/1 and 10/5 bps, fraction 1 portfolio, mark audit and 240m reconciliation. Exploratory shorter holding hypotheses motivated by report 23; not optimized or independent, does not establish dynamic strategy or retrained horizon performance. Implemented in existing portfolio_engine.py (ec09fd4) and CLI (16e2512). Runtime unverified. No new Python files, downloads, training or holdout.

```cmd
git pull origin main
python research_pipeline.py time-exit
```
Upload data/reports/36_time_exit.zip even if failed.


## Checkpoint — Report 36 PASS; retain 4h baseline
36_time_exit.zip inspected: C01 120/180/240m primary expectancy (%) .343197/.389682/.551195, PF 1.91735/1.81705/2.32961, compounded return (%) 40.73674/47.01195/74.03352, MTM DD (%) 6.36830/10.51520/12.79002, Sharpe .98740/1.04938/1.59919; stress return (%) 16.71944/21.93658/44.36823. C02 corresponding expectancy (%) .523778/.523823/.690801, PF 2.35780/2.15953/2.80220, return (%) 37.41547/37.14455/52.46708, DD (%) 6.66905/10.51520/12.79002, Sharpe .97700/.97104/1.42300; stress return (%) 22.68933/22.44221/36.13117. Same 104/63 entries. Park shorter unconditional exits for tuning; retain as lower-DD alternatives, not evidence against retrained shorter targets. Next Report 37 early invalidation: two fixed checkpoint alternatives 60/120m after entry, if signed gross return at checkpoint minute open <=0, exit at next minute open; otherwise original 240m exit. Fixed original entries, no re-entry, no training, same cost scenarios and common calendar; mark coverage and 240m baseline reconciliation. Exploratory choices motivated by report 23; no independent validation. Existing portfolio_engine.py (58121fc) and unified CLI (eb68efd), runtime unverified. No downloads, new Python files or holdout. After report 37, close initial exit screening unless material evidence, then OBV/filter technical benchmark.

```cmd
git pull origin main
python research_pipeline.py early-invalidation
```
Upload data/reports/37_early_invalidation.zip even if failed.


## Checkpoint — reports 37–38; incremental transfer
Report 37 earlier user conversation: early invalidation 60/120m inferior to original 4h for C01/C02; retain 4h and park tested exit rules. Report 38 inspected directly: OBV20/100 and filter1%/2% each 6047 trades, zero unknown funding exclusions; standard net expectancy (%) -.113417/-.088475/-.095020/-.113219, PF .740494/.790320/.777480/.740327; stress expectancy (%) -.293453/-.268517/-.275051/-.293247. C01 reconciles 104 trades, +.551195% expectancy, PF 2.329612, fixed P&L 57.324242%. These are fixed 4h direction adaptations, not stateful execution or exact Deprez & Frömmel replication. Park four tested configurations, retain stateful OBV/filter as deferred hypothesis. Next Report 39 incremental independent per-symbol retraining of frozen C01 on SOL/TRX/LTC/NEAR/AAVE/FIL/WAVES/OMG, subject to full-grid assertions and local data. Existing transfer engine (cfde280) and CLI (1ec3140), no new Python files; original nine altcoin fits not rerun. BTC cached comparator; same 42 folds, 11 technical features, 1h/4h/15m, C and seed, .70 threshold. Sequential processing and fold checkpoints. Preliminary fee/slippage-only economics; no funding or mark repair until screening supports it. Listing/delisting verification pending; full-window selection bias; no holdout/tuning. Source code changes alter new experiment hashes; existing caches remain intact. Runtime unverified.

```cmd
git pull origin main
python research_pipeline.py extended-transfer
```
Upload data/reports/39_extended_transfer.zip even if partial or failed.


## Checkpoint — Report 39 PARTIAL; prioritize TRX and NEAR
Directly inspected 39_extended_transfer.zip: seven additional assets completed, WAVES failed minimum train/test sample assertion (not a negative economic result). SOL/TRX/LTC/NEAR/AAVE/FIL/OMG standard fee/slippage-only expectancy (%) -.036548/.694704/.083839/.715735/.448388/.323111/.730031; PF .984/1.916/1.090/1.431/1.298/1.288/1.463. All seven Brier scores worse than naive. TRX 118 trades, primary fixed P&L 81.975%, DD 20.362%, stress P&L 60.620%; 2023 negative, 62 trades in 2024, nominal 7/30d expectancy CI crosses zero even primary. NEAR promising but DD 40.507%. Prioritize TRX and NEAR for funding audit; park SOL/LTC, defer AAVE/FIL and OMG (contract lifecycle unverified). WAVES eligibility diagnosis deferred, preserve cached work. No new confirmed candidate before funding and risk validation. Next Report 40 reuses multi_asset_funding_engine.py (da87d73), CLI (7e8dd78), reads frozen report 39 predictions, strict 8h funding schedule gate; nonstandard schedules require separate evidence. No downloads, training, new Python files or holdout. Runtime unverified.

```cmd
git pull origin main
python research_pipeline.py extended-funding
```
Upload data/reports/40_extended_funding.zip even if partial or failed.


## Checkpoint — Report 40 PASS; provisional C07/C08
Direct inspection of 40_extended_funding.zip: TRX/NEAR each 4383 standard 8h events, 21 missing settlement marks each, zero affected trades. Primary net expectancy (%) .694988/.733511, PF 1.914721/1.442375, fixed-notional P&L (%) 82.008601/63.815458, realized DD (%) 20.459392/40.200466. Stress P&L (%) 60.652787/47.979301. TRX nominal CIs cross zero at both costs; NEAR standard 7d CI [.000877%,1.554607%], 30d [.148612%,1.404997%], stress both cross zero. NEAR annual P&L positive each year even stress; TRX 2023 negative. No multiple-testing adjustment. Add provisional C07 TRX and C08 NEAR, not tuning priorities. Next Report 41 uses existing portfolio_engine.py (2f2e642), symbol-aware mark windows and frozen report39/40 reconciliation, funding source hash and schedule checks, no dropped trades, independent per-symbol replay at .25/.5/1 notional fractions, standard/stress costs. Not a combined portfolio. Missing intratrade marks block only affected symbol and are reported for targeted repair. Minute-open proxies, no margin/liquidation, no independent holdout. CLI fcb3eea. Runtime unverified; no new Python files, training or downloads. Continue broader hypothesis screening after this bounded risk audit.

```cmd
git pull origin main
python research_pipeline.py alt-portfolio
```
Upload data/reports/41_alt_portfolio.zip even if failed.


## Checkpoint — Report 41 PASS; static sleeve diversification
Initial report 41 blocked by source hash path mismatch; funding engine relative path fix 0cf990c and report 40 rescore resolved it. Uploaded 41_alt_portfolio(1).zip inspected: TRX/NEAR mark coverage 28320/20880 minutes, zero affected trades. Standard compounded returns (%) at .25/.5/1 fractions TRX 22.0427/47.2796/107.8081, NEAR 16.1013/32.1149/61.3475; MTM DD (%) TRX 8.9984/17.1430/31.2308, NEAR 12.3742/23.7120/43.4894. At fraction 1 standard Sharpe .64704/.72382, stress return (%) 68.0376/37.7981 and MTM DD 32.3500/45.4628. BTC C01 remains incumbent; TRX/NEAR retain diversification-only priority, not individual tuning. Next Report 42: equal initial capital sleeves BTC alone, BTC+TRX, BTC+NEAR, BTC+TRX+NEAR, no rebalancing, same fraction per sleeve .25/.5/1, common calendar and standard/stress costs. Reuse daily equity and ledgers from reports 35/41, compute correlations, overlap minutes, daily-close drawdown, daily Sharpe, terminal sleeve return contribution. No combined intraday MTM or shared margin modeled; not optimized or independent. Implemented in portfolio_engine.py e61830d and CLI 2b32aa4; runtime unverified. No training, downloads or new Python files.

```cmd
git pull origin main
python research_pipeline.py diversification
```
Upload data/reports/42_diversification.zip even if failed.


## Checkpoint — Report 42 PASS; close equal-sleeve screening
Inspected uploaded 42_diversification.zip: at fraction 1, standard return (%) BTC/BTC+TRX/BTC+NEAR/all three = 74.033522/90.920832/67.690504/81.063050; daily-close DD (%) 5.303505/15.134000/11.039058/11.172647; daily Sharpe 1.599190/.915438/1.297554/1.066965; stress return (%) 44.368226/56.202934/41.083162/50.067989. BTC has highest Sharpe and lowest daily-close DD across all tested fractions and costs. Daily strategy return correlations BTC/TRX .071602, BTC/NEAR -.039637, TRX/NEAR .008784 at standard fraction 1; overlap minutes 2220/6060/3180. Park tested equal initial capital sleeve combinations; no inference against all possible diversified allocations. C07/C08 remain evidence, not tuning priorities. No combined intraday MTM, no independent validation.
Next Report 43 closes deferred stateful OBV/filter hypothesis from Report 38: reuse identical OBV20/100 and percentage filter 1/2% signals in technical_rule_engine.py (cd88310), hold until signal changes, neutral flatten, opposite reverse, 15m delay, terminal close, common incumbent eligible grid, report decision gaps and exposure. Recovered funding ledger, no unknown-funding exclusions allowed, standard/stress costs, fixed-notional screening only. No exact paper replication, no tuning, new files, download or training. CLI 4ed39dd. Runtime unverified. Then proceed to research integrity and bounded remaining literature hypotheses before nested tuning.

```cmd
git pull origin main
python research_pipeline.py stateful-extended
```
Upload data/reports/43_stateful_extended.zip, or terminal traceback if blocked.


## Checkpoint — Report 44 PASS; frozen prediction reproducibility
Uploaded 44_research_integrity.zip directly inspected: status pass, 42 folds and 48 monthly feature checks, dataset_hash_matches true, current and original SHA256 both 06e9533c3b29d33a567e4f40d9164188c7adf705cc366a9a075e166b0fb14600. Existing report 44 checks passed for manifest, feature finite-value reconstruction, raw entry/exit prices, timing, purge, common eligibility, probabilities, training priors, funding hash and cost-aware targets. Not an exhaustive leakage or prediction reproducibility proof. Report 43 prior conversation recorded stateful OBV/filter screening as parked; its numerical results not reverified in this checkpoint. Next Report 45 refits only frozen C01/C02 on the same 42 folds, sequentially, same model parameters and preprocessing; compares maximum absolute probability error <=1e-6 and zero threshold signal mismatches. No original checkpoint overwrites, tuning, downloads or holdout. CLI reproduce implemented in research_pipeline.py f9e98f3. Runtime unverified. Refit is intentional and limited to reproducibility, not a new experiment search.

```cmd
git pull origin main
python research_pipeline.py reproduce
```
Upload data/reports/45_reproducibility.zip even if failed.


## Checkpoint — Report 45 PASS; conditional predictive diagnostics
Uploaded 45_reproducibility.zip inspected directly: 42 folds, 84 refits total, maximum absolute probability error exactly zero for both C01 and C02, zero threshold signal mismatches. Frozen configurations reproducible in same environment. No further identical refits required. This does not establish independent predictive skill or economic robustness.
Next Report 46: descriptive probability reliability in fixed decile bins, class-specific observed frequency, Brier sum versus train-fold prior, conditional target and direction hit rates, net win rate and expectancy at existing .70 threshold, for C01/C02. All qualifying signal opportunities before non-overlap selection; overlapping outcomes not independent trades and not portfolio results. No calibration fitting, threshold optimization, new model, training, download or holdout. C01 binary direction probabilities and C02 three-class cost-aware probabilities are not directly comparable via raw Brier sum. Implemented in research_pipeline.py eb8ab4a; runtime unverified. Remaining literature hypotheses are not declared exhausted; defer resource-heavy branches pending evidence and data. Nested tuning and multiple-testing controls remain outstanding.

```cmd
git pull origin main
python research_pipeline.py predictive-diagnostics
```
Upload data/reports/46_predictive_diagnostics.zip even if failed.


## Checkpoint — Report 46 PASS; calibration comparator
Verified uploaded report 46: 30071 OOS observations each. C01 Brier sum .5003892783 versus prior .4998822425; C02 .6186405196 versus prior .6233061394 (different targets; not directly comparable). At threshold .70, C01 long/short signals 149/24, mean probability .765358/.733542, actual target frequency .657718/.583333, net expectancy .00326543/.00475678. C02 long/short signals 91/12, mean probability .776026/.747620, target frequency .626374/.5, net expectancy .00372404/.00090090. All four groups show descriptive overconfidence; short sample sparse and signals overlap, no inference of statistical significance or realized portfolio performance. Retain both candidates and both sides.
Report 47: prequential single-temperature scaling, expanding prior completed OOS folds only, first six folds calibration warmup, evaluate remaining 36 on common calendar; compare raw vs calibrated Brier/log loss and non-overlap economic results at unchanged .70 threshold, standard/stress costs. Temperature optimized for prior log loss on [.25,8]; no threshold fitting, no model retraining, no holdout. This is an exploratory calibration algorithm comparison, not independent evidence; reduced signal count at fixed probability threshold may be expected and is not alone proof of economic inferiority. Existing CLI research_pipeline.py d067c1d. Runtime unverified.

```cmd
git pull origin main
python research_pipeline.py calibration
```
Upload data/reports/47_calibration.zip even if failed.
