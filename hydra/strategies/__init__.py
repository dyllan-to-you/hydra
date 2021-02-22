from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from hydra.types import Price
from typing import List, NamedTuple, Tuple, TypedDict, Union
import pandas as pd


class Decision(Enum):
    BUY = 1
    SELL = -1
    NONE = 0


class DecisionEvent(TypedDict):
    timestamp: datetime
    decision: Decision


class Strategy(ABC):
    decision_history: List[DecisionEvent]
    last_decision: Decision

    def __init__(self):
        super().__init__()
        self.decision_history = []
        self.last_decision = Decision.SELL
        self.name = ""

    @property
    def decision_history_df(self):
        df = pd.DataFrame(self.decision_history, columns=DecisionEvent._fields)
        df.set_index("timestamp", inplace=True)
        return df

    def decide(self, price_history, *args, **kwargs) -> Tuple[Decision, float]:
        decision, decision_price = self._decide(price_history, *args, **kwargs)
        # if decision is not Decision.NONE:
        #     self.decision_history.append(
        #         DecisionEvent(price_history[-1]["Date"], decision)
        #     )

        # if decision == Decision.BUY or decision == Decision.SELL:
        #     print(decision, price_history)

        if decision == Decision.NONE:
            return Decision.NONE, 0

        if (self.last_decision == Decision.SELL and decision == Decision.BUY) or (
            self.last_decision == Decision.BUY and decision == Decision.SELL
        ):
            self.last_decision = decision
            self.decision_history.append(
                {
                    "timestamp": price_history[-1]["Date"],
                    "decision": decision,
                    "price": decision_price,
                }
            )

            return decision, decision_price
        return Decision.NONE, 0

    @abstractmethod
    def _decide(self, price_history) -> Tuple[Decision, float]:
        pass

    @abstractmethod
    def init_indicators(self):
        pass