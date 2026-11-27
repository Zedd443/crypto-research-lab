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

The active search specification is centralized in `research_engine.py` (`SPEC` and grid constants). The initial grid is 30m/1h/4h, 1/4/8/24h target horizon, binary and cost-aware logistic targets, threshold .55/.60/.65/.70, fixed exit or half-horizon invalidation, coarse C=1 then C=.1/10 on the best inner family. Delay=15m and rolling training=180d fixed initially. 120d inner fit / 60d inner validation; outer 30d test. Choose only from past validation; no forced trade. Prediction fits are reused for thresholds and exits. Model/data/selection identities preserve completed compatible checkpoints. The initial grid is not exhaustive and does not establish a validated edge.

## Evaluation scope

Signal discovery only. All qualifying long/short opportunities are retained, including simultaneous signals across assets. Per-symbol nonoverlap is a separate diagnostic to avoid counting repeated overlapping outcomes as independent evidence; it is not a capital constraint. Inner selection requires positive standard and stress expectancy on that diagnostic and ranks stress expectancy shrunk toward zero (ten zero-return pseudo-observations, an initial research design choice). No minimum trading-frequency quota. Report pooled and per-symbol/per-side/per-fold results, equal-symbol mean expectancy and nominal calendar-block bootstrap for nonoverlap outcomes. Fixed-unit return sums are not portfolio returns.

Fee, slippage and funding are included. Settlement marks remain required for known funding costs; unknown outcomes stay in the signal ledger and known-only statistics are explicitly conditional. No intratrade mark replay, margin, position sizing, capital allocation, leverage or liquidation checks in this phase. One ZIP includes metrics.json and signals.json for later allocation research. Portfolio and production work are deferred until signal candidates pass independent evidence gates.

## Data

UTC; end exclusive; immutable verified monthly raw Parquet. Reuse existing files, source-keyed caches and fold checkpoints. Download/repair only data needed by a defined hypothesis. Legacy command history is preserved in the roadmap, not the active workflow.
