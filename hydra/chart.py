from hydra import Hydra, AroonStrategy, Price
import os
from csv import DictReader
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import matplotlib.dates as mpl_dates

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
            float(row.get("open")),
            float(row.get("high")),
            float(row.get("low")),
            float(row.get("close")),
            float(row.get("Volume BTC")),
            float(row.get("Volume USD")),
        )
        hydra.add_head(price)

    data = pd.DataFrame.from_dict(hydra.price_history)
    data = data.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume BTC",
            "volume_usd": "Volume",
        }
    )

    data["Date"] = pd.to_datetime(data["timestamp"])
    data = data.drop(["Volume BTC", "timestamp"], axis=1)
    data.set_index("Date", inplace=True)
    print(list(data.columns))
    print(data)
    mpf.plot(data.iloc[::-1], type="candle", volume=True)

# def test_version():
#     assert __version__ == '0.1.0'
