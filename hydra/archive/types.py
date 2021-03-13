from enum import Enum
from typing import NamedTuple, TypedDict
from datetime import datetime


class Price(TypedDict):
    Date: datetime
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float
    Volume_USD: float