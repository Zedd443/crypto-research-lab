# Crypto Research Lab

Binance USD-M futures research focused on finding robust, cost-aware signal edge across assets. Portfolio allocation and production execution follow only after signal validation. No validated live edge or live order execution yet. See [active roadmap](docs/ROADMAP.md).

## Active workflow (Windows CMD)

From the repository root with the existing .venv activated:

```cmd
git pull origin main
python -m pytest -q
python research_pipeline.py batch
```

The batch runs nested signal searches and frozen baseline comparisons automatically, retaining all qualifying signals. One upload after the whole run: `data/reports/research_batch.zip` (also if status is partial/fail). If tests fail before the batch, send the traceback. Do not run historical commands one by one. Repeat the same command after interruption; completed per-symbol/fold checkpoints are reused. No automatic downloads or data repair, no holdout access.

Default pilot: BTCUSDT ETHUSDT BNBUSDT TRXUSDT NEARUSDT (previously screened, funding audited). Optional `--symbols` accepts a subset of the known crypto universe. Full development raw/funding grid required for now; symbols with incomplete coverage or nonstandard funding are blocked, not imputed. Listing-aware dynamic universes are a later gate.

## Source map

| Role | Active modules |
|---|---|
| User entry point | research_pipeline.py |
| Data acquisition and raw aggregation | data_engine.py, derivatives_engine.py, universe.py |
| Feature definitions | feature_engine.py |
| Joint parameter search and signal-edge evaluation | research_engine.py |
| Chronological validation and resumable identities | validation_protocol.py, experiment_store.py, resource_policy.py |
| Transaction costs / settlement funding marks | execution_engine.py, robustness_engine.py, funding_repair_engine.py |
| Existing predictive comparators | baseline_engine.py |
| Unit tests | tests/ |

All other standalone experiment/repair/report engines are historical evidence or optional specialized utilities, not steps the user needs to run individually. Keep them in place for historical imports and hashes until a separately tested migration. Do not delete raw data, old reports or checkpoints. The new batch creates one cohesive active module rather than another script per parameter.

## Signal discovery protocol

The search specification is centralized in `research_engine.py` (`SPEC` and grid constants). Current search: 30m/1h/4h timeframe, 1/4/8/24h prediction horizon, binary and cost-aware targets, logistic and depth-3 decision tree, probability thresholds .55/.60/.65/.70; fixed exit or half-horizon invalidation. Logistic C=1 coarse then .1/10 on its best inner family. Non-ML comparators: momentum/reversal on ret_4/ret_16 with train-only absolute return quantiles .80/.95. This is bounded, not exhaustive; no inference against untested model families.

Fixed 15m delay, 180d outer train, two forward 30d inner validation windows following 120d/150d training; outer 30d test. All training labels end before the next evaluation window; inner comparison uses common maximum-horizon end and known funding eligibility. No outer outcome used as an entry filter. Binary training needs price labels; cost-aware training additionally needs verified costs, so target samples can differ. Reuse predictions for threshold and exit variants, vectorized inner outcomes, source-keyed base cache and per-symbol/fold checkpoints. Inner selection ranks positive standard-cost nonoverlap expectancy shrunk toward zero (ten zero-return pseudo-observations, a heuristic), with stress costs separately reported rather than a hard veto. No trading-frequency quota.

Report fixed historical baseline, nested overall selection and independently inner-selected logistic/tree/rule families on outer development folds; matched same-time equal-random-direction comparator; per-symbol, per-fold, per-side and pooled expectancy, win rate, PF, coverage, equal-symbol mean and nominal calendar-block bootstrap. Matched direction comparator uses the same entry time and exit policy for the opposite direction; it does not isolate timing skill. No model wins solely because it has better predictive accuracy. Historical outer folds remain development evidence, not untouched holdout. Multiple-testing control and fresh evaluation are outstanding. A completed batch is not proof of edge.

All qualifying long/short opportunities are retained across assets. Within-symbol nonoverlap is a diagnostic, not shared capital allocation. Fee, slippage and funding included. Unknown funding outcomes stay in the ledger, with conditional known-only statistics and partial status; no intratrade mark replay, margin, sizing, capital allocation or liquidation checks. One ZIP includes metrics.json and signals.json. Do not infer portfolio P&L from fixed-unit outcome sums.

## Data

UTC; end exclusive; immutable verified monthly raw Parquet. Reuse existing files, source-keyed caches and fold checkpoints. Download/repair only data needed by a defined hypothesis. Legacy command history is preserved in the roadmap, not the active workflow.
