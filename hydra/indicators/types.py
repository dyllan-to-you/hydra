from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Dict, List, NamedTuple, Tuple, TypedDict
from pyarrow import Field


class Decision(Enum):
    BUY = 1
    SELL = -1


class Trade(TypedDict):
    decision: Decision
    timestamp: datetime
    price: float


class Indicator(ABC):
    name: str
    fields: List[Tuple[str]]
    tier: int = 0

    @abstractmethod
    def calc(self, price, history) -> NamedTuple:
        pass

    def calculate(self, price, history) -> Dict[str, Dict]:
        output = self.calc(price, history)
        return {self.name: output._asdict()}
