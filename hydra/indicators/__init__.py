from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, NamedTuple, TypedDict
from hydra.strategies import Decision


class Trade(TypedDict):
    decision: Decision
    timestamp: datetime
    price: float


class Indicator(ABC):
    name: str
    tier: int

    def __init__(self, tier=0):
        self.tier = tier

    @abstractmethod
    def calc(self, price, history) -> Dict:
        pass

    def calculate(self, price, history) -> Dict[str, Dict]:
        output = self.calc(price, history)
        return {self.name: output}
