from datetime import datetime
from enum import Enum
from typing import NamedTuple


class Direction(int, Enum):
    BUY = 1
    SELL = 0


class BuyOrder(NamedTuple):
    timestamp: datetime
    trigger_price: float
    simulation_id: int
    direction: Direction = Direction.BUY


class SellOrder(NamedTuple):
    timestamp: datetime
    trigger_price: float
    profit: float
    direction: Direction = Direction.SELL
