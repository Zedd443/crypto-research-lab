# Crypto Research Lab

Binance USD-M futures research. Separate from Replay Lab. Initial E00a implementation: 30-contract pilot, 1m archives, SHA256 verification, partitioned Parquet, gap reports and UTC resampling. No validated trading strategy yet.

## Windows

```powershell
git clone https://github.com/Zedd443/crypto-research-lab.git
cd crypto-research-lab
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
python data_engine.py download --symbols BTCUSDT --start 2024-01-01 --end 2024-02-01
python data_engine.py resample --symbols BTCUSDT
```

Dates UTC, end exclusive. Default download covers all 30 configured symbols from September 2019 through latest closed minute; large job. Missing dates may be prelisting, delisted or unavailable; not imputed. Existing monthly partitions are reused if complete for requested range. Derivatives, listing-aware archive inventory, external copy and native candle comparison are pending E00b. See docs/ROADMAP.md.
