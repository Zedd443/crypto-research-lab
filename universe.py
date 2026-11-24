GROUPS = {
    "large_cap": "BTC ETH BNB SOL XRP DOGE ADA TRX LINK AVAX BCH LTC DOT UNI ATOM NEAR HBAR SUI AAVE FIL".split(),
    "delisted": "WAVES XEM ORBS LOOM OMG".split(),
    "listed_2024": "WIF ENA PNUT VIRTUAL PENGU".split(),
}
UNIVERSE = {k: [s + "USDT" for s in v] for k,v in GROUPS.items()}
SYMBOLS = [s for v in UNIVERSE.values() for s in v]
TIMEFRAMES = {"5m":"5m","15m":"15m","30m":"30m","1h":"1h","4h":"4h","8h":"8h","12h":"12h","1d":"1d","1w":"1w"}
