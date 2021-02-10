import os
import pytest
from csv import DictReader
from typing import cast
from unittest.mock import patch
import dateutil.parser as parser
from hydra import Hydra
from hydra.types import Price
import hydra.indicators.HeikinAshi as HeikinAshi
import mplfinance as mpf


@pytest.mark.skip(reason="Not Focused")
def test_heikin_ashi_chart():
    with open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../data/Bitfinex_BTCUSD_1h.test.csv",
        )
    ) as file:
        hydra = Hydra(HeikinAshi.Indicator())
        prices = DictReader(file)
        for idx, row in enumerate(prices):
            if idx > 8760:
                break
            price = Price(
                parser.parse(cast(str, row.get("date"))),
                float(cast(str, row.get("open"))),
                float(cast(str, row.get("high"))),
                float(cast(str, row.get("low"))),
                float(cast(str, row.get("close"))),
                float(cast(str, row.get("Volume BTC"))),
                float(cast(str, row.get("Volume USD"))),
            )
            hydra.feed(price)

        data = hydra.history_df
        data.set_index("Date", inplace=True)

        filter_col = [col for col in data if col.startswith(HeikinAshi.NAME)]
        print(filter_col)
        ha_data = data[filter_col].rename(lambda n: n.split(".")[-1], axis=1)
        print(data, ha_data)
        subplots = [
            mpf.make_addplot(ha_data, type="candle", panel=1),
        ]

        mpf.plot(data, type="candle", addplot=subplots, panel_ratios=(1, 1))
