import os
from abc import ABC
from csv import DictReader
from typing import List, NamedTuple, Tuple
from enum import Enum
from datetime import datetime
import numpy as np


class Decision(Enum):
    BUY = 1
    SELL = -1


class Trade(NamedTuple):
    decision: Decision
    timestamp: datetime
    price: float


class Price(NamedTuple):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    volume_usd: float


class Strategy(ABC):
    required_history: int

    def should_buy(history=None) -> bool:
        pass

    def should_sell(history=None, last_trade=None) -> bool:
        pass


class AroonStrategy(Strategy):
    required_history = 25

    def __init__(self, period=25):
        self.period = period
        self.required_history = period

    def aroon(self, history):
        period = self.period if len(history) >= self.period else len(history)
        timespan = history[:period]
        min_idx = np.argmin([getattr(price, "low") for price in timespan])
        max_idx = np.argmax([getattr(price, "high") for price in timespan])

        up = 100 * (period - max_idx) / period
        down = 100 * (period - min_idx) / period
        oscillation = up - down

        return oscillation, up, down


trade_history: List[Trade] = []


class Hydra:
    price_history: List[Price] = []
    last_decision: Decision = Decision.SELL

    aroon_history: List = []

    def __init__(self, strategy: Strategy):
        self.strategy = strategy

    def add_head(self, price):
        self.price_history.insert(0, price)
        self.aroon_history.insert(0, self.strategy.aroon(self.price_history))
        # self.price_history = self.price_history[: self.strategy.required_history]

    def decision(self):
        if self.last_decision == Decision.SELL:
            if self.strategy.should_buy(self.price_history):
                trade_history.append((Decision.BUY,))
            return
        if self.last_trade.decision == Decision.BUY:
            if self.strategy.should_sell(
                self.price_history, last_trade=trade_history[-1]
            ):
                trade_history.append((Decision.SELL,))


def main():
    with open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../data/Bitfinex_BTCUSD_1h.csv",
        )
    ) as file:
        hydra = Hydra(AroonStrategy())
        prices = DictReader(file)
        for row in prices:
            price = Price(
                row.get("date"),
                float(row.get("open")),
                float(row.get("high")),
                float(row.get("low")),
                float(row.get("close")),
                float(row.get("Volume BTC")),
                float(row.get("Volume USD")),
            )
            hydra.add_head(price)
            hydra.decision()