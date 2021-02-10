from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from hydra.indicators.types import Indicator
import hydra.indicators.Aroon as Aroon
from typing import Dict, List, NamedTuple, Union


class Decision(Enum):
    BUY = 1
    SELL = -1
    NONE = 0


class DecisionEvent(NamedTuple):
    timestamp: datetime
    decision: Decision


class Strategy(ABC):
    history: List[DecisionEvent]

    def decide(self, history, *args, **kwargs) -> Decision:
        decision = self._decide(history, *args, **kwargs)
        if decision is not Decision.NONE:
            self.decisions.append(DecisionEvent(history[-1]["Date"], decision))
        return decision

    @abstractmethod
    def _decide(self, history) -> Decision:
        pass

    @abstractmethod
    def init_indicators(self):
        pass


class AroonStrategy(Strategy):
    indicator: Indicator

    def __init__(self, period=25):
        self.indicator = Aroon.Indicator(period)

    def init_indicators(self):
        return self.indicator

    def _decide(self, history) -> Decision:
        try:
            last_oscillator = history[-2][f"{self.indicator.name}.oscillator"]
            this_oscillator = history[-1][f"{self.indicator.name}.oscillator"]
            if last_oscillator < 0 and this_oscillator >= 0:
                return Decision.BUY

            if last_oscillator > 0 and this_oscillator <= 0:
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