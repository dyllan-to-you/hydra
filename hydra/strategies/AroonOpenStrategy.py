from typing import Tuple
from numpy import NaN
from . import Strategy
from hydra.indicators import Aroon, AroonOpen, Indicator, Decision
import pandas as pd


class AroonOpenStrategy(Strategy):
    indicator: Indicator

    def __init__(self, indicator=Aroon.Indicator, period=25):
        super().__init__()
        self.period = period
        self.indicator = indicator(period)
        self.open_indicator = AroonOpen.Indicator(period)
        self.name = f"AroonOpenStrategy({period})"

    def init_indicators(self):
        return [self.indicator, self.open_indicator]

    def check_aroon(self, indicator, price_history) -> Tuple[Decision, float]:
        try:
            this_price = price_history[-1]
            # print(indicator.name, this_price)
            this_oscillator = this_price[indicator.name]["oscillator"]
            last_oscillator = price_history[-2][indicator.name]["oscillator"]
            if last_oscillator <= 0 and this_oscillator > 0:
                return self.pick_price(Decision.BUY, this_price)

            if last_oscillator >= 0 and this_oscillator < 0:
                return self.pick_price(Decision.SELL, this_price)
        except IndexError:
            pass

        return None

    def _decide(self, price_history) -> Tuple[Decision, float]:

        if len(price_history) <= self.period + 1:
            return (Decision.NONE, NaN)

        check_open = self.check_aroon(self.open_indicator, price_history)
        if check_open is not None:
            return check_open

        check_normal = self.check_aroon(self.indicator, price_history)
        if check_normal is not None:
            return check_normal

        return (Decision.NONE, NaN)

    def pick_price(self, decision, price) -> Tuple[Decision, float]:
        if decision == Decision.BUY:
            if price[self.indicator.name]["up"] < 100:
                return decision, price["Open"]
            return decision, price[self.indicator.name]["peak"]

        if decision == Decision.SELL:
            if price[self.indicator.name]["down"] < 100:
                return decision, price["Open"]
            return decision, price[self.indicator.name]["valley"]
