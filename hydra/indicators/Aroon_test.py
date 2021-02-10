import os
from csv import DictReader
from typing import cast
import dateutil.parser as parser
from numpy.lib import math
from hydra.types import Price
import hydra.indicators.Aroon as Aroon
import mplfinance as mpf
from hydra import Hydra
import pytest
from unittest.mock import patch


@pytest.mark.skip(reason="Excel Spreadsheet needs correcting")
def test_aroon():
    with open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../data/Bitfinex_BTCUSD_1h.test.csv",
        )
    ) as file:
        history = []
        aroon = Aroon.Indicator()
        prices = DictReader(file)
        for idx, row in enumerate(prices):
            price = Price(
                parser.parse(cast(str, row.get("date"))),
                float(cast(str, row.get("open"))),
                float(cast(str, row.get("high"))),
                float(cast(str, row.get("low"))),
                float(cast(str, row.get("close"))),
                float(cast(str, row.get("Volume BTC"))),
                float(cast(str, row.get("Volume USD"))),
            )._asdict()
            period, timespan = aroon.get_timespan(price, history)
            assert period == min(25, idx + 1)
            assert len(timespan) == period

            min_idx, max_idx = aroon.get_indexes(timespan)
            assert min_idx == int(cast(str, row.get("Aroon D Idx")))
            assert max_idx == int(cast(str, row.get("Aroon Up Idx"))), f"Row {idx}"
            # TODO: Current error is due to excel spreadsheet not getting "Last Match"
            calc = aroon.calc(price, history)
            assert calc.down == round(
                float(cast(str, row.get("Aroon Down")))
            ), f"Row {idx}"
            assert calc.up == round(
                float(cast(str, (row.get("Aroon Up"))))
            ), f"Row {idx}"
            assert calc.oscillator == round(
                float(cast(str, (row.get("Aroon Oscillator"))))
            ), f"Row {idx}"

            calculate = aroon.calculate(price, history)
            history.insert(0, price | calculate)


# @pytest.mark.skip(reason="Not Focused")
def test_aroon_chart():
    btcvol = "Volume BTC"

    with open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../data/Bitfinex_BTCUSD_1h.test.csv",
        )
    ) as file:
        indicators = [Aroon.Indicator((i + 1) * 5) for i, x in enumerate([0] * 9)]
        hydra = Hydra(indicators)
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
                float(cast(str, row.get(btcvol))),
                float(cast(str, row.get("Volume USD"))),
            )
            hydra.feed(price)

        data = hydra.history_df
        data = data.rename(
            columns={
                "Volume": btcvol,
                "Volume_USD": "Volume",
            }
        )

        data.set_index("Date", inplace=True)
        charts = []
        for idx, indicator in enumerate(indicators):
            charts.append(
                mpf.make_addplot(
                    data[f"{indicator.name}.up"], color="green", panel=idx + 1
                )
            )
            charts.append(
                mpf.make_addplot(
                    data[f"{indicator.name}.down"], color="red", panel=idx + 1
                )
            )
        # aroon = [
        #     mpf.make_addplot(data[f"aroon.up"], color="green", panel=1),
        #     mpf.make_addplot(data["aroon.down"], color="red", panel=1),
        #     # mpf.make_addplot(
        #     #     data["aroon.oscillator"],
        #     #     ylabel="Aroon Oscillator",
        #     #     color="grey",
        #     #     panel=2,
        #     #     secondary_y=True,
        #     # ),
        #     # mpf.make_addplot(
        #     #     data[btcvol],
        #     #     panel=1,
        #     #     color="yellow",
        #     #     ylabel="BTC Vol.",
        #     #     linestyle="dotted",
        #     #     secondary_y="auto",
        #     # ),
        # ]

        mpf.plot(data, type="candle", datetime_format="%Y-%m-%d %H:%M", addplot=charts)
