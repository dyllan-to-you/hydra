import numpy as np
import copy
import pandas as pd
from typing import List, NamedTuple, TypedDict
from . import Indicator as AIndicator
from hydra.types import Price


class Output(TypedDict):
    up: float
    down: float
    oscillator: float
    peak: float
    valley: float


NAME = "aroon_open"


class Indicator(AIndicator):
    period: int
    tier: int

    def __init__(self, period=25, **kwargs):
        self.period = period
        self.name = f"aroon_open({self.period})"
        super().__init__(**kwargs)

    def get_indexes(self, history_segment):
        reversed = history_segment[::-1]
        lows = [p["Low"] for p in reversed]
        highs = [p["High"] for p in reversed]
        min_idx = np.argmin(lows)
        max_idx = np.argmax(highs)
        return min_idx, max_idx

    def get_line(self, period, idx) -> float:
        a = period - idx
        b = a / period
        return b * 100

    def get_history_segment(self, price, history):
        period: int = self.period if len(history) >= self.period else len(history) + 1
        openPrice = price["Open"]
        new_price = {"High": openPrice, "Low": openPrice}
        history_segment = (history + [new_price])[-period - 1 :]
        return period, history_segment

    # def get_last_history_segment(self, history):
    #     period: int = self.period if len(history) > self.period else len(history)
    #     last_history_segment = (history)[-period:-1]
    #     return last_history_segment

    def calc(self, price, history: List[Price]) -> Output:
        period, history_segment = self.get_history_segment(price, history)
        min_idx, max_idx = self.get_indexes(history_segment)
        max = history_segment[-max_idx - 1]
        min = history_segment[-min_idx - 1]

        up = self.get_line(period, max_idx)
        down = self.get_line(period, min_idx)
        oscillator = up - down

        return {
            "up": round(up),
            "down": round(down),
            "oscillator": round(oscillator),
            "peak": max["High"],
            "valley": min["Low"],
        }
