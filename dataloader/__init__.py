from json import load
import pandas as pd
from binance import Client

from dataloader.pairs import binance_to_kraken
from . import binance_data
from . import kraken


def load_prices(
    b_pair,
    startDate=None,
    endDate=None,
    interval=Client.KLINE_INTERVAL_1MINUTE,
):
    k_pair = binance_to_kraken(b_pair)
    k_interval = (
        int(pd.to_timedelta(interval).total_seconds() / 60)
        if isinstance(interval, str)
        else interval
    )
    assert (
        k_interval <= 1440
    ), "This is to make sure that 1m isn't interpreted as a month"

    b_data = binance_data.load_prices(b_pair, startDate, endDate, interval)

    if k_pair is None:
        return b_data

    k_data = kraken.load_prices(k_pair, startDate, endDate, k_interval)

    # if len(b_data) == 0:
    #     b_data = binance_data.load_prices(
    #         b_pair, startDate, endDate, interval, log=True
    #     )
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