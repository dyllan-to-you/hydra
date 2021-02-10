from enum import Enum
from typing import NamedTuple
from datetime import datetime


class Price(NamedTuple):
    Date: datetime
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float
    Volume_USD: float