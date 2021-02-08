import os
from csv import DictReader
import dateutil.parser as parser
import mplfinance as mpf
from typing import cast
from hydra import Hydra
from hydra.types import Price
import hydra.indicators.Aroon as Aroon

btcvol = "Volume BTC"

with open(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../data/Bitfinex_BTCUSD_1h.csv",
    )
) as file:
    hydra = Hydra([Aroon.Indicator()])
    prices = DictReader(file)
    for idx, row in enumerate(prices):
        if idx > 100:
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
            "volume": btcvol,
            "volume_usd": "Volume",
        }
    )

    data.set_index("Date", inplace=True)

    print(data)
    aroon = [
        mpf.make_addplot(
            data[btcvol],
            panel=1,
            color="yellow",
            ylabel="BTC Vol.",
            linestyle="dotted",
            secondary_y="auto",
        ),
        mpf.make_addplot(data["aroon.up"], color="green", panel=2),
        mpf.make_addplot(data["aroon.down"], color="red", panel=2),
        mpf.make_addplot(
            data["aroon.oscillator"],
            ylabel="Aroon Oscillator",
            color="grey",
            panel=2,
            secondary_y=True,
        ),
    ]

    mpf.plot(data.iloc[::-1], type="candle", volume=True, addplot=aroon)

# def test_version():
#     assert __version__ == '0.1.0'
