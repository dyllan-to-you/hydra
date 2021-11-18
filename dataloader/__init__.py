from json import load
import pandas as pd
from binance import Client
from . import binance_data
from . import kraken


# pairmap = {
#     "XBTUSD": {
#         "kraken": "XBTUSD",
#         "binance": "BTCUSD"
#     }
# }


def load_prices(
    b_pair,
    k_pair=None,
    startDate=None,
    endDate=None,
    interval=Client.KLINE_INTERVAL_1MINUTE,
):
    k_interval = (
        int(pd.to_timedelta(interval).total_seconds() / 60)
        if isinstance(interval, str)
        else interval
    )

    b_data = binance_data.load_prices(b_pair, startDate, endDate, interval)

    if k_pair is None:
        return b_data

    k_data = kraken.load_prices(k_pair, startDate, endDate, k_interval)

    if len(b_data) == 0:
        b_data = binance_data.load_prices(
            b_pair, startDate, endDate, interval, log=True
        )
    prices = b_data.combine_first(k_data)
    if len(prices) == 0:
        print(startDate, endDate)
        print("PRICE LEN IS 0 WAT")
        print(b_data)
        print(k_data)
    return prices


if __name__ == "__main__":
    load_prices(
        None,
        "2019-01-01",
        "2020-12-31",
    )
