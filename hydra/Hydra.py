import pandas as pd
from pandas.core.frame import DataFrame
from hydra.types import Price
import itertools
from typing import Dict, List, Set, Tuple, Union
from hydra.strategies import Strategy
from hydra.indicators import Decision, Indicator
from hydra.utils import flatten


class Hydra:
    price_history: List[Dict]
    decisions: List[Decision]
    # Key is priority, Value is List of Indicators
    indicators: Dict[int, List[Indicator]]
    indicator_set: Set
    strategy: Strategy

    def __init__(
        self,
        strategy: Strategy,
        indicators: Union[List[Indicator], Indicator] = None,
        name: str = "",
        fee: float = 0,
    ):
        self.price_history = []
        self.decisions = []
        self.indicators = {}
        self.fee = fee
        self.name = name
        self.indicator_set = set()
        self.set_strategy(strategy)
        self.add_indicators(indicators)

    def set_strategy(self, strategy):
        # Get dependent indicators w/parameters & set for Hydra
        self.add_indicators(strategy.init_indicators())
        self.strategy = strategy

    def add_indicators(
        self,
        indicators: Union[List[Indicator], Indicator],
    ):
        if isinstance(indicators, list):
            for indicator in indicators:
                self.add_indicator(indicator)
        elif indicators is not None:
            self.add_indicator(indicators)

    def add_indicator(self, indicator: Indicator):
        if indicator.name not in self.indicator_set:
            self.indicators.setdefault(indicator.tier, []).append(indicator)
        return self

    @property
    def prioritized_indicators(self):
        return itertools.chain(*self.indicators.values())

    @property
    def price_history_df(self) -> DataFrame:
        df = pd.json_normalize(self.price_history)
        df.set_index("Date", inplace=True)
        return df

    # add new price
    def feed(self, food: Price) -> Tuple[Tuple[Decision, float], Price]:
        # price: Price = {
        #     "Date": food["Date"],
        #     "Open": food["Open"],
        #     "High": food["High"],
        #     "Low": food["Low"],
        #     "Close": food["Close"],
        #     "Volume": food["Volume"],
        #     "Volume_USD": food["Volume_USD"],
        # }
        price = food
        for indicator in self.prioritized_indicators:
            price |= indicator.calculate(price, self.price_history)
        # print("\n", self.strategy.indicator.name, pd.json_normalize(self.price_history))
        self.price_history.append(price)
        # print("\n", self.strategy.indicator.name, pd.json_normalize(self.price_history))

        decision = self.strategy.decide(self.price_history)

        return decision, price