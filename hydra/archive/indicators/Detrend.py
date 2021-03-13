import numpy as np
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


NAME = "aroon"


class Indicator(AIndicator):
    period: int
    tier: int
    max: float
    max_idx: int
    min: float
    min_idx: int

    def __init__(self, period=25, **kwargs):
        self.period = period
        self.name = f"aroon({self.period})"
        self.last_max = self.last_max_idx = None
        self.last_min = self.last_min_idx = None
        super().__init__(**kwargs)

    def get_indexes(self, price):
        if price["High"] >= self.last_max:
            max_idx = 0
        else:
            max_idx = self.last_max_idx + 1

        if price["Low"] <= self.last_min:
            min_idx = 0
        else:
            min_idx = self.last_min_idx + 1
        
        return min_idx, max_idx

    def get_line(self, period, idx) -> float:
        return (period - idx) / period * 100

    def get_history_segment(self, history):
        period: int = self.period if len(history) >= self.period else len(history) + 1
        history_segment = history[-period:]
        return period, history_segment

    # def get_last_history_segment(self, history):
    #     period: int = self.period if len(history) > self.period else len(history)
    #     last_history_segment = (history)[-period:-1]
    #     return last_history_segment

    def calc(self, this_price, history: List[Price]) -> Output:
        period, history_segment = self.get_history_segment(history)
        # max = history_segment[-(max_idx + 1)]["High"]
        # min = history_segment[-(min_idx + 1)]["Low"]



        if len(history_segment) > 0:
            last_price = history_segment[-1]
            if self.last_max is None or last_price["High"] >= self.last_max:
                self.last_max = last_price["High"]
                self.last_max_idx = 0
            else:
                self.last_max_idx += 1
                if self.last_max_idx >= period:
                    self.last_max_idx = np.argmax([p["High"] for p in history_segment[::-1]])
                    self.last_max = history_segment[-(self.last_max_idx + 1)]["High"]
                    # print('max', max_idx, self.last_max_idx, max,  self.last_max, pd.json_normalize(history_segment))

            if self.last_min is None or last_price["Low"] <= self.last_min:
                self.last_min = last_price["Low"]
                self.last_min_idx = 0
            else:
                self.last_min_idx += 1
                if self.last_min_idx >= period:
                    self.last_min_idx = np.argmin([p["Low"] for p in history_segment[::-1]])
                    self.last_min = history_segment[-(self.last_min_idx + 1)]["Low"]
                    # print('min', min_idx, self.last_min_idx, min, self.last_min, pd.json_normalize(history_segment))
        else:
            self.last_max = this_price["High"]
            self.last_min = this_price["Low"]
            self.last_min_idx = 0
            self.last_max_idx = 0


        min_idx, max_idx = self.get_indexes(this_price)

        up = self.get_line(period, max_idx)
        down = self.get_line(period, min_idx)
        oscillator = up - down

        return {
            "up": up,
            "down": down,
            "oscillator": oscillator,
            "peak": self.last_max,
            "valley": self.last_min,
        }
