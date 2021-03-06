import math
import sys
import time
from datetime import datetime
from hydra.strategies import Decision, DecisionEvent
import numpy as np
import pandas as pd
from typing import List, NamedTuple, TypedDict, cast
import dateutil.parser as parser
from hydra import Hydra
from hydra.types import Price
from hydra.strategies.PSARoonStrategy import PSARoonStrategy
from tqdm import tqdm
import pyarrow.parquet as pq
import pathlib
from datetime import datetime
import hydra.BokehChart as bokeh
import sqlite3
from datetime import datetime
from numba import njit
import numba
import vectorbt as vbt
from vectorbt.signals.factory import SignalFactory

vbt.settings.ohlcv["column_names"] = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}


@njit
def aroon_entry(from_i, to_i, col, a, temp_idx_arr):
    if from_i == 0:
        w = np.where(a[:, col] > 99)[0]
        for i, num in enumerate(w):
            temp_idx_arr[i] = num

    for i in range(len(temp_idx_arr)):
        if temp_idx_arr[i] > from_i and temp_idx_arr[i] < to_i:
            return temp_idx_arr[i : i + 1]
    return temp_idx_arr[:0]


@njit
def aroon_exit(from_i, to_i, col, a, temp_idx_arr):
    if temp_idx_arr[-1] != 42:
        temp_idx_arr[-1] = 42
        w = np.where(a[:, col] < -99)[0]
        for i, num in enumerate(w):
            temp_idx_arr[i] = num

    for i in range(len(temp_idx_arr)):
        if temp_idx_arr[i] > from_i and temp_idx_arr[i] < to_i:
            return temp_idx_arr[i : i + 1]
    return temp_idx_arr[:0]


def run_sim(
    pairs,
    batches,
    startDate="2020-01-01",
    endDate="2021-01-01",
    interval=1,
    path="./data/kraken/",
):
    res = []
    for pair in pairs:
        filepath = pathlib.Path().absolute()
        parqpath = filepath.joinpath(path, "parq", f"{pair}_{interval}.parq")
        table = pq.read_table(parqpath)
        prices = table.to_pandas()
        prices["time"] = pd.to_datetime(prices["time"], unit="s")
        prices = prices.set_index("time").loc[startDate:endDate]  # ["close"]
        prices.drop("trades", axis=1, inplace=True)
        print("Prices:", prices)

        vbt.settings.portfolio["freq"] = f"{interval}m"
        vbt.settings.portfolio["init_cash"] = 100.0  # in $
        vbt.settings.portfolio["fees"] = 0.0006  # in %
        # vbt.settings.portfolio["slippage"] = 0.0025  # in %
        # vbt.settings.caching["blacklist"].append("Portfolio")
        # vbt.settings.caching["whitelist"].extend(
        #     ["Portfolio.cash_flow", "Portfolio.share_flow"]
        # )

        for batch in batches:
            AROONOSC = vbt.IndicatorFactory.from_talib("AROONOSC")
            # print(help(AROONOSC.run))

            aroonosc = AROONOSC.run(
                prices["high"], prices["low"], **batch.get("AROONOSC")
            )
            SAREXT = vbt.IndicatorFactory.from_talib("SAREXT")
            # print(help(SAREXT.run))
            sarext = SAREXT.run(prices["high"], prices["low"], **batch.get("SAREXT"))

            # Build signal generator
            AroonStrategy = SignalFactory(input_names=["aroon"]).from_choice_func(
                entry_choice_func=aroon_entry,
                entry_settings=dict(
                    pass_inputs=["aroon"],
                    pass_kwargs=["temp_idx_arr"],  # built-in kwarg
                ),
                exit_choice_func=aroon_exit,
                exit_settings=dict(
                    pass_inputs=["aroon"],
                    pass_kwargs=["temp_idx_arr"],  # built-in kwarg
                ),
                forward_flex_2d=True,
            )
            # Run strategy signal generator
            aroon_signals = AroonStrategy.run(aroonosc.real)

            portfolio = vbt.Portfolio.from_signals(
                prices["close"], aroon_signals.entries, aroon_signals.exits
            )

            # with pd.option_context(
            #     "display.max_rows", None, "display.max_columns", None
            # ):  # more options can be specified also

            #     print(
            #         portfolio.stats(agg_func=None)
            #         # portfolio.orders.buy.count(),
            #         # portfolio.orders.sell.count(),
            #     )

            # portfolio.value().vbt.plot()

            # res.append(portfolio)

    return res


psars = [(x * 0.0002) + 0.01 for x in range(50)]

t0 = time.time()
portfolios = run_sim(
    ["XBTUSD"],
    batches=[
        {
            "AROONOSC": {"timeperiod": list(range(2, 10))},
            "SAREXT": {"startvalue": [0.01]},
        }
    ],
    startDate="2018-05-15",
    endDate="2021-07-15",
    interval=1,
)

t1 = time.time()
print("Total Time Elapsed:", t1 - t0)
# for portfolio in portfolios:
# print(portfolio.total_return())
# print(portfolio.total_profit())
