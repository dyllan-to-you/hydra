from . import Strategy
from hydra.indicators import Indicator, Decision, Aroon
import pandas as pd


class AroonStrategy(Strategy):
    indicator: Indicator

    def __init__(self, indicator, period=25):
        super().__init__()
        self.period = period
        self.indicator = indicator(period)
        self.name = f"AroonStrategy({period})"

    def init_indicators(self):
        return self.indicator

    def _decide(self, price_history) -> Decision:

        if len(price_history) <= self.period + 1:
            return Decision.NONE

        try:
            this_oscillator = price_history[-1][self.indicator.name]["oscillator"]
            last_oscillator = price_history[-2][self.indicator.name]["oscillator"]
            if last_oscillator <= 0 and this_oscillator > 0:
                return Decision.BUY

            if last_oscillator >= 0 and this_oscillator < 0:
                return Decision.SELL
        except IndexError:
            pass

        return Decision.NONE


# class MetaStrategy(Strategy):
#     strategies: Union[List[Strategy], Dict[str, Strategy]]

#     def decide(self, history) -> Decision:
#         decisions = {}
#         for strat in self.strategies:
#             decisions[strat.name] = strat.decide()

#         # Weigh Decisions based on other factors
#         # ???
#         # Profit
#         decision = None
#         self.history.append(decision)
#         return decision