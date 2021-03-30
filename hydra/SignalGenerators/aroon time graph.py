import duckdb
import pathlib
import numpy as np
from numba import njit
import vectorbt as vbt
from vectorbt.signals.factory import SignalFactory
from hydra.SuperSim import load_prices
import pyarrow as pa
import pyarrow.parquet as pq
from hydra.utils import timeme, now
from time import time
import os, psutil

process = psutil.Process(os.getpid())

name = "Aroon"


@njit
def aroon_entry(from_i, to_i, col, a, temp_idx_arr):
    if from_i == 0:
        w = np.where(a[:, col] > 50)[0]
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
        w = np.where(a[:, col] < -50)[0]
        for i, num in enumerate(w):
            temp_idx_arr[i] = num

    for i in range(len(temp_idx_arr)):
        if temp_idx_arr[i] > from_i and temp_idx_arr[i] < to_i:
            return temp_idx_arr[i : i + 1]
    return temp_idx_arr[:0]


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

AROONOSC = vbt.IndicatorFactory.from_talib("AROONOSC")
# printd(help(AROONOSC.run))


def generate_signals(prices, timeperiods):
    aroonosc = AROONOSC.run(prices["high"], prices["low"], timeperiods)
    # Run strategy signal generator
    aroon_signals = AroonStrategy.run(aroonosc.real)
    return aroon_signals


@timeme
def main(
    pair,
    path,
    startDate,
    endDate,
    interval,
):
    prices = load_prices(pair, path, startDate, endDate, interval)
    print("time,", "aroon_size")
    for i in range(0, 100):
        b4 = process.memory_info().rss / 1024
        timeperiods = list(range(100, 101 + i))
        size = len(timeperiods)
        ts = time()
        signals = generate_signals(prices, timeperiods)
        te = time()
        print(
            te - ts, ",", size, ",", b4, ",", process.memory_info().rss / 1024
        )  # in bytes


main(
    pair="XBTUSD",
    path="../data/kraken",
    startDate="2017-05-15",
    endDate="2021-05-16",
    interval=1,
)
