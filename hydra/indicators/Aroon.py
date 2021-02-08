import numpy as np
import pandas as pd
from typing import List, NamedTuple
from .types import Indicator as AIndicator
from hydra.types import Price


class Output(NamedTuple):
    up: float
    down: float
    oscillator: float


class Indicator(AIndicator):
    name = "aroon"
    period: int

    def __init__(self, period=25):
        self.period = period

    def get_indexes(self, timespan):
        min_idx = np.argmin([p["Low"] for p in timespan])
        max_idx = np.argmax([p["High"] for p in timespan])
        return min_idx, max_idx

    def get_line(self, period, idx) -> float:
        return ((period - idx) / period) * 100

    def get_timespan(self, price, history):
        period: int = self.period if len(history) >= self.period else len(history) + 1
        timespan = ([price] + history)[:period]
        return period, timespan

    def calc(self, price, history: List[Price]) -> Output:
        period, timespan = self.get_timespan(price, history)
        min_idx, max_idx = self.get_indexes(timespan)

        up = self.get_line(period, max_idx)
        down = self.get_line(period, min_idx)
        oscillator = up - down

        return Output(round(up), round(down), round(oscillator))
