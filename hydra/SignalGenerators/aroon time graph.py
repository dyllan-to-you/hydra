import gc
import numpy as np
from numba import njit
import vectorbt as vbt
from vectorbt.signals.factory import SignalFactory
from hydra.SuperSim import load_prices
from hydra.utils import timeme, now
from hydra.SignalGenerators.AroonSignalGenerator import generate_signals
from time import time
import os, psutil

process = psutil.Process(os.getpid())


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
        b4 = process.memory_info().rss / 1024 / 1024
        timeperiods = list(range(100, 101 + i))
        size = len(timeperiods)
        ts = time()
        signals = generate_signals(prices, timeperiods)
        gc.collect()
        te = time()
        print(
            f"{te - ts:.2f}, {size}, {b4:.2f}, "
            f"{process.memory_info().rss /1024/1024 : .2f}"
        )  # in bytes


main(
    pair="XBTUSD",
    path="../data/kraken",
    startDate="2017-05-15",
    endDate="2021-05-16",
    interval=1,
)
