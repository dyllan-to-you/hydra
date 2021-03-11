import pickle
import time
from datetime import datetime, date
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
import vectorbt as vbt
from vectorbt.signals.factory import SignalFactory

# vbt.settings.caching["blacklist"].append("Portfolio")
# vbt.settings.caching["whitelist"].extend(
#     ["Portfolio.cash_flow", "Portfolio.share_flow"]
# )

def printd(*arg):
    print(date.today().strftime("[%Y-%m-%d %H:%M:%S]"), *arg)



def get_methods(object, spacing=20):
  methodList = []
  for method_name in dir(object):
    try:
        if callable(getattr(object, method_name)):
            methodList.append(str(method_name))
    except:
        methodList.append(str(method_name))
  processFunc = (lambda s: ' '.join(s.split())) or (lambda s: s)
  for method in methodList:
    try:
        print(str(method.ljust(spacing)) + ' ' +
              processFunc(str(getattr(object, method).__doc__)[0:90]))
    except:
        print(method.ljust(spacing) + ' ' + ' getattr() failed')

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
    name=None
):
    vbt.settings.ohlcv["column_names"] = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    for pair in pairs:
        printd('Simulating', pair)
        filepath = pathlib.Path().absolute()
        parqpath = filepath.joinpath(path, f"{pair}_{interval}.parq")
        table = pq.read_table(parqpath)
        prices = table.to_pandas()
        prices["time"] = pd.to_datetime(prices["time"], unit="s")
        prices = prices.set_index("time").loc[startDate:endDate]  # ["close"]
        prices.drop("trades", axis=1, inplace=True)
        # printd("Prices:", prices)
        printd(pair, 'prices loaded')

        for count, batch in enumerate(batches):
            printd(pair, 'batch', count, ': Generating Indicators')

            AROONOSC = vbt.IndicatorFactory.from_talib("AROONOSC")
            # printd(help(AROONOSC.run))
            aroonosc = AROONOSC.run(
                prices["high"], prices["low"], **batch.get("AROONOSC")
            )

            SAREXT = vbt.IndicatorFactory.from_talib("SAREXT")
            # printd(help(SAREXT.run))
            sarext = SAREXT.run(prices["high"], prices["low"], **batch.get("SAREXT"))
            printd(pair, 'batch', count, ': Generated Indicators')

            printd(pair, 'batch', count, ': Generating Strategy')
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
                # forward_flex_2d=True,
            )
            # Run strategy signal generator
            aroon_signals = AroonStrategy.run(aroonosc.real)
            printd(pair, 'batch', count, ': Generated Strategy')

            printd(pair, 'batch', count, ': Simulating Orders')
            portfolio = vbt.Portfolio.from_signals(
                prices["close"], aroon_signals.entries, aroon_signals.exits,
                freq = f"{interval}m",
                init_cash = 100.0,  # in $
                fees = 0.0006,  # in %
                # slippage = 0.0025  # in %
            )

            # printd(portfolio.sharpe_ratio())
            # portfolio.save('portfolio_config')
            # portfolio = vbt.Portfolio.load('portfolio_config')
            # portfolio.sharpe_ratio()            # printd(pair, 'batch', count, ': Simulated Orders')

            # printd('Portfolio Methods:', get_methods(portfolio))
            # printd(issubclass(type(portfolio), vbt.Pickleable))

            printd(pair, 'batch', count, ': Saving file')

            # startvalue=0, offsetonreverse=0, accelerationinitlong=0.02, accelerationlong=0.02, accelerationmaxlong=0.2, accelerationinitshort=0.02, accelerationshort=0.02, accelerationmaxshort=0.2
            filename = f"{pair} Aroon {batch['AROONOSC']['timeperiod'][0]}-{batch['AROONOSC']['timeperiod'][-1]}.portfolio"
            if name is not None:
                filename = f"{name} {filename}"
            portfolio.save(filename)
            # pickle.dump( portfolio, open( f"{filename}.p", "wb" ) )




def start():
    t0 = time.time()
    portfolios = run_sim(
        ["XBTUSD"],
        batches=[
            {
                "AROONOSC": {"timeperiod": list(range(100, 101))},
                "SAREXT": {"startvalue": [0.01]},
            }
        ],
        startDate="2018-05-15",
        endDate="2018-07-15",
        interval=1,
    )

    t1 = time.time()
    printd("Total Time Elapsed:", t1 - t0)
    # for portfolio in portfolios:
    # printd(portfolio.total_return())
    # printd(portfolio.total_profit())
