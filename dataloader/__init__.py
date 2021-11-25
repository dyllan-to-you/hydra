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
    interval=1,
):
    k_pair = binance_to_kraken(b_pair)

    b_data = binance_data.load_prices(b_pair, startDate, endDate, interval)

    if k_pair is None:
        return b_data

    k_data = kraken.load_prices(k_pair, startDate, endDate, interval)

    # if len(b_data) == 0:
    #     b_data = binance_data.load_prices(
    #         b_pair, startDate, endDate, interval, log=True
    #     )
    # print("----kraken----", k_data)
    # print("++++binance+++", b_data)
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
