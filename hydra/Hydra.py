import pandas as pd
from hydra.types import Price
import itertools
from typing import Dict, List
from hydra.indicators.types import Indicator
from hydra.utils import flatten


class Hydra:
    history: List[Dict] = []
    # Key is priority, Value is List of Indicators
    indicators: Dict[int, List[Indicator]] = {}
    # Prioritised List
    _heads: List[Indicator] = []

    def __init__(self, indicators: List[Indicator]):
        for indicator in indicators:
            self.add_head(indicator)

    def add_head(self, indicator: Indicator):
        self.indicators.setdefault(indicator.tier, []).append(indicator)
        return self

    @property
    def heads(self):
        return itertools.chain(*self.indicators.values())

    @property
    def history_df(self):
        return pd.json_normalize(self.history)

    # add new price
    def feed(self, food):
        price = food._asdict()
        for head in self.heads:
            price |= head.calculate(price, self.history)

        self.history.insert(0, price)

        return self
