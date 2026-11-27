# Crypto Research Lab

Binance USD-M futures research toward a shared-capital, multi-asset trading system. No validated live edge or live order execution yet. See [active roadmap](docs/ROADMAP.md).

## Active workflow (Windows CMD)

From the repository root with the existing .venv activated:

```cmd
git pull origin main
python -m pytest -q
python research_pipeline.py batch
```

The batch runs all planned nested searches, frozen baseline comparisons and shared-wallet replays automatically. One upload after the whole run: `data/reports/research_batch.zip` (also if status is partial/fail). If tests fail before the batch, send the traceback. Do not run historical commands one by one. Repeat the same command after interruption; completed per-symbol/fold checkpoints are reused. No automatic downloads or data repair, no holdout access.

Default pilot: BTCUSDT ETHUSDT BNBUSDT TRXUSDT NEARUSDT (previously screened, funding audited). Optional `--symbols` accepts a subset of the known crypto universe. Full development raw/funding grid required for now; symbols with incomplete coverage or nonstandard funding are blocked, not imputed. Listing-aware dynamic universes are a later gate.

## Source map

| Role | Active modules |
|---|---|
| User entry point | research_pipeline.py |
| Data acquisition and raw aggregation | data_engine.py, derivatives_engine.py, universe.py |
| Feature definitions | feature_engine.py |
| Joint search, strategy selection and shared wallet | research_engine.py |
| Chronological validation and resumable identities | validation_protocol.py, experiment_store.py, resource_policy.py |
| Shared transaction costs / funding and mark lookup | execution_engine.py, robustness_engine.py, funding_repair_engine.py, portfolio_engine.py |
| Existing predictive comparators | baseline_engine.py |
| Unit tests | tests/ |

All other standalone experiment/repair/report engines are historical evidence or optional specialized utilities, not steps the user needs to run individually. Keep them in place for historical imports and hashes until a separately tested migration. Do not delete raw data, old reports or checkpoints. The new batch creates one cohesive active module rather than another script per parameter.

The active search specification is centralized in `research_engine.py` (`SPEC` and grid constants). The initial grid is 30m/1h/4h, 1/4/8/24h target horizon, binary and cost-aware logistic targets, threshold .55/.60/.65/.70, fixed exit or half-horizon invalidation, coarse C=1 then C=.1/10 on the best inner family. Delay=15m and rolling training=180d fixed initially. 120d inner fit / 60d inner validation; outer 30d test. Choose only from past validation; no forced trade. Prediction fits are reused for thresholds and exits. Model/data identities are separate from wallet-only changes. New batch is not a claim that the best global parameter or optimal portfolio has been found.

Shared-wallet comparison uses one normalized unlevered equity, fixed 5 slots, target 20% equity entry notional, 95% entry gross reservation cap, minimum 5% partial allocation, one position per symbol, past-validation ranking, fee/slippage/funding and minute-open MTM. This is an exposure model, not actual Binance margin tiers, leverage, liquidation or fill guarantees. Unknown mark/funding inputs block affected replay rather than silently discard trades. All results remain exploratory; no live orders.

## Data

UTC; end exclusive; immutable verified monthly raw Parquet. Reuse existing files, source-keyed caches and fold checkpoints. Download/repair only data needed by a defined hypothesis. Legacy command history is preserved in the roadmap, not the active workflow.
