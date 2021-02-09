import pandas as pd
from pandas.core.frame import DataFrame
from hydra.types import Price
import itertools
from typing import Dict, List, Union
from hydra.indicators.types import Indicator
from hydra.utils import flatten


class Hydra:
    history: List[Dict] = []
    # Key is priority, Value is List of Indicators
    indicators: Dict[int, List[Indicator]] = {}

    def __init__(self, indicators: Union[List[Indicator], Indicator]):
        if isinstance(indicators, list):
            for indicator in indicators:
                self.add_indicator(indicator)
        else:
            self.add_indicator(indicators)

    def add_indicator(self, indicator: Indicator):
        self.indicators.setdefault(indicator.tier, []).append(indicator)
        return self

    @property
    def prioritized_indicators(self):
        return itertools.chain(*self.indicators.values())

    @property
    def history_df(self) -> DataFrame:
        return pd.json_normalize(self.history)

    # add new price
    def feed(self, food):
        price = food._asdict()
        for indicator in self.prioritized_indicators:
            price |= indicator.calculate(price, self.history)

        self.history.append(price)

        return self
