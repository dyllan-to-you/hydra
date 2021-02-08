import os
from csv import DictReader
from typing import cast
import dateutil.parser as parser
from numpy.lib import math
from tqdm import tqdm
from hydra.types import Price
import hydra.indicators.Aroon as Aroon


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
