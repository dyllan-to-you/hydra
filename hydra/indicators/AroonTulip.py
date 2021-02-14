import numpy as np
import pandas as pd
from typing import List, NamedTuple, TypedDict
from . import Indicator as AIndicator
from hydra.types import Price
import tulipy
from inspect import getmembers, isfunction


def print_info(indicator):
    print("Type:", indicator.type)
    print("Full Name:", indicator.full_name)
    print("Inputs:", indicator.inputs)
    print("Options:", indicator.options)
    print("Outputs:", indicator.outputs)


class Output(TypedDict):
    up: float
    down: float
    oscillator: float
    peak: float
    valley: float


NAME = "aroonTA"


class Indicator(AIndicator):
    period: int
    tier: int

    def __init__(self, period=25, **kwargs):
        self.period = period
        self.name = f"aroonTA({self.period})"
        super().__init__(**kwargs)

    # def get_indexes(self, history_segment):
    #     reversed = history_segment[::-1]
    #     min_idx = np.argmin([p["Low"] for p in reversed])
    #     max_idx = np.argmax([p["High"] for p in reversed])
    #     return min_idx, max_idx

    def get_line(self, period, idx) -> float:
        return ((period - idx) / period) * 100

    def get_history_segment(self, price, history):
        period: int = self.period if len(history) >= self.period else len(history) + 1
        history_segment = (history + [price])[-period - 1 :]
        return period, history_segment

    def get_last_history_segment(self, history):
        period: int = self.period if len(history) > self.period else len(history)
        last_history_segment = (history)[-period:]
        return last_history_segment

    def calc(self, price, history: List[Price]) -> Output:
        period, history_segment = self.get_history_segment(price, history)
        if len(history_segment) <= period:
            return {}
        df = pd.DataFrame(history_segment)
        down, up = tulipy.aroon(
            df["High"].to_numpy(), df["Low"].to_numpy(), period=period
        )

        oscillator = up - down
        # real = tulipy.aroonosc(
        #     df["High"].to_numpy(), df["Low"].to_numpy(), period=period
        # )
        # assert (oscillator - real) < 0.00001

        last_history_segment = self.get_last_history_segment(history)
        dfl = pd.DataFrame(last_history_segment)
        if len(dfl) == 0:
            return None
        valley = np.amin(dfl["Low"])
        peak = np.amax(dfl["High"])
        return {
            "up": round(up[-1]),
            "down": round(down[-1]),
            "oscillator": round(oscillator[-1]),
            "peak": peak,
            "valley": valley,
        }
