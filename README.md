# Crypto Research Lab

Local-first Binance USD-M futures research, separate from Crypto Replay Lab. Phase E00 builds verified historical datasets; strategies and profitability are not yet validated.

## Windows setup

```powershell
git clone https://github.com/Zedd443/crypto-research-lab.git
cd crypto-research-lab
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python data_engine.py --help
python data_engine.py download --symbols BTCUSDT --start 2024-01-01 --end 2024-02-01
python data_engine.py resample --symbols BTCUSDT
python -m pytest -q
```

Dates are UTC, end exclusive. Download all 30 configured contracts with `python data_engine.py download` (large job). Monthly archive preferred, daily fallback, REST for recent missing days; no API keys. Missing archives are reported, never filled with synthetic candles. Monthly Parquet partitions retain the original ten kline fields plus close time. Existing files are validated before reuse; rerun to repair missing intervals. The default start is September 2019; actual coverage depends on available listing history. Delisting boundaries are discovered from archive availability and exchange metadata where possible; absent dates are not proof of delisting. Resampling drops incomplete buckets and uses UTC boundaries (Monday for weeks). The old Replay Lab is not modified.

Additional derivatives, spot and external data support is planned; see docs/ROADMAP.md. No live trading.
