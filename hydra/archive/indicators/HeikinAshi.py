import numpy as np
import pandas as pd
from typing import List, NamedTuple, TypedDict
from . import Indicator as AIndicator
from hydra.types import Price

NAME = "heikin-ashi"


class Output(TypedDict):
    Close: float
    Open: float
    High: float
    Low: float


class Indicator(AIndicator):
    def __init__(self, period=1, **kwargs):
        self.period = period
        self.name = f"{NAME}({self.period})"
        super().__init__(**kwargs)

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

        return {"Close": close, "Open": open, "High": high, "Low": low}
