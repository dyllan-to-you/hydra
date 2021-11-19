pairmap = {"BTCUSD": "XBTUSD"}


def binance_to_kraken(pair):
    return pairmap.get(pair, pair)