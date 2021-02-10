import numpy as np
import pandas as pd
from typing import List, NamedTuple
from . import Indicator as AIndicator
from hydra.types import Price


class Output(NamedTuple):
    up: float
    down: float
    oscillator: float


NAME = "aroon"


class Indicator(AIndicator):
    period: int
    tier: int

    def __init__(self, period=25, **kwargs):
        self.period = period
        self.name = f"aroon({self.period})"
        super().__init__(**kwargs)

    def get_indexes(self, timespan):
        reversed = timespan[::-1]
        min_idx = np.argmin([p["Low"] for p in reversed])
        max_idx = np.argmax([p["High"] for p in reversed])
        return min_idx, max_idx

    def get_line(self, period, idx) -> float:
        return ((period - idx) / period) * 100

    def get_timespan(self, price, history):
        period: int = self.period if len(history) >= self.period else len(history) + 1
        timespan = (history + [price])[-period:]
        return period, timespan

    def calc(self, price, history: List[Price]) -> Output:
        period, timespan = self.get_timespan(price, history)
        min_idx, max_idx = self.get_indexes(timespan)

        up = self.get_line(period, max_idx)
        down = self.get_line(period, min_idx)
        oscillator = up - down

        return Output(round(up), round(down), round(oscillator))
