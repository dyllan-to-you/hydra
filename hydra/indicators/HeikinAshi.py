import numpy as np
import pandas as pd
from typing import List, NamedTuple
from .types import Indicator as AIndicator
from hydra.types import Price

NAME = "heikin-ashi"


class Output(NamedTuple):
    Close: float
    Open: float
    High: float
    Low: float


class Indicator(AIndicator):
    name = NAME

    def __init__(self, period=1):
        self.period = period

    def calc(self, price: Price, history: List[Price]) -> Output:
        try:
            previous = history[-1][self.name]
        except IndexError:
            previous = price
            pass

        close = (price["Open"] + price["Close"] + price["High"] + price["Low"]) / 4
        open = (previous["Open"] + previous["Close"]) / 2

        high = max(price["High"], open, close)
        low = min(price["Low"], open, close)

        return Output(close, open, high, low)
