GROUPS = {
    "large_cap": "BTC ETH BNB SOL XRP DOGE ADA TRX LINK AVAX BCH LTC DOT UNI ATOM NEAR HBAR SUI AAVE FIL".split(),
    "delisted": "WAVES XEM ORBS LOOM OMG".split(),
    "listed_2024": "WIF ENA PNUT VIRTUAL PENGU".split(),
}
UNIVERSE = {k: [s + "USDT" for s in v] for k,v in GROUPS.items()}
CRYPTO_SYMBOLS = [s for v in UNIVERSE.values() for s in v]
TRADFI_SYMBOLS = ["XAUUSDT", "QQQUSDT", "SPYUSDT", "NVDAUSDT", "AAPLUSDT", "CLUSDT", "BZUSDT"]
UNIVERSE["tradfi"] = TRADFI_SYMBOLS
SYMBOLS = CRYPTO_SYMBOLS + TRADFI_SYMBOLS
# Conservative UTC start dates from official launch announcements.
TRADFI_START = {"XAUUSDT":"2026-01-05","QQQUSDT":"2026-04-06","SPYUSDT":"2026-04-06","NVDAUSDT":"2026-03-26","AAPLUSDT":"2026-04-06","CLUSDT":"2026-04-01","BZUSDT":"2026-04-01"}
TIMEFRAMES = {"5m":"5m","15m":"15m","30m":"30m","1h":"1h","4h":"4h","8h":"8h","12h":"12h","1d":"1d","1w":"1w"}
