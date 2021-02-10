from hydra.Strategy import Strategy
import pandas as pd
from pandas.core.frame import DataFrame
from hydra.types import DecisionEvent, Price
import itertools
from typing import Dict, List, Set, Union
from hydra.indicators.types import Decision, Indicator
from hydra.utils import flatten


class Hydra:
    history: List[Dict] = []
    decisions: List[Decision] = []
    # Key is priority, Value is List of Indicators
    indicators: Dict[int, List[Indicator]] = {}
    indicator_set: Set
    strategy: Strategy

    def __init__(
        self,
        strategy: Strategy,
        indicators: Union[List[Indicator], Indicator],
    ):
        self.set_strategy(strategy)
        self.add_indicators(indicators)

    def set_strategy(self, strategy):
        # Get dependent indicators w/parameters & set for Hydra
        self.add_indicators(strategy.init_indicators())
        pass

    def add_indicators(
        self,
        indicators: Union[List[Indicator], Indicator],
    ):
        if isinstance(indicators, list):
            for indicator in indicators:
                self.add_indicator(indicator)
        else:
            self.add_indicator(indicators)

    def add_indicator(self, indicator: Indicator):
        if indicator.name not in self.indicator_set:
            self.indicators.setdefault(indicator.tier, []).append(indicator)
        return self

    @property
    def prioritized_indicators(self):
        return itertools.chain(*self.indicators.values())

    @property
    def history_df(self) -> DataFrame:
        return pd.json_normalize(self.history)

    # add new price
    def feed(self, food: Price):
        price = food._asdict()
        for indicator in self.prioritized_indicators:
            price |= indicator.calculate(price, self.history)

        self.history.append(price)

        decision = self.strategy.decide(self.history)

        return decision