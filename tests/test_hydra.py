from hydra import Hydra, AroonStrategy, Price
import os
from csv import DictReader
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import pandas as pd
import matplotlib.dates as mpl_dates

plt.style.use("ggplot")


def test_chart():
    with open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../data/Bitfinex_BTCUSD_1h.csv",
        )
    ) as file:
        hydra = Hydra(AroonStrategy())
        prices = DictReader(file)
        for row in prices:
            price = Price(
                row.get("date"),
                row.get("open"),
                row.get("high"),
                row.get("low"),
                row.get("close"),
                row.get("Volume BTC"),
                row.get("Volume USD"),
            )
            hydra.add_head(price)


# def test_version():
#     assert __version__ == '0.1.0'
