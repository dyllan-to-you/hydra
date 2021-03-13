import numpy as np
import pandas as pd
from typing import List, TypedDict

from pygments.token import Error
from . import Indicator as AIndicator
from hydra.types import Price


class Output(TypedDict):
    PSAR: float
    PSARextreme: float
    direction: str
    PStARt: float


NAME = "PSAR"


class Indicator(AIndicator):
    tier: int

    def __init__(self, AFstart=0.02, AFstep=0.02, AFmax=0.2, start_period=5, **kwargs):
        self.AFstart = AFstart
        self.AFstep = AFstep
        self.AFmax = AFmax
        self.AF = AFstart
        self.start_period = start_period
        self.name = f"PSAR({self.AFstart}, {self.AFstep}, {self.AFmax}, {start_period})"
        self.PSARextreme = None
        super().__init__(**kwargs)

    def get_starting_PSAR(self, history, this_price) -> Output:
        history_segment = history[-self.start_period :]
        if history_segment[0]["Close"] > history_segment[4]["Close"]:
            self.direction = "FALLING"
            return {
                "PSAR": max([p["High"] for p in history_segment]),
                "PSARextreme": this_price["Low"],
                "direction": self.direction,
            }
        else:
            self.direction = "RISING"
            return {
                "PSAR": min([p["Low"] for p in history_segment]),
                "PSARextreme": this_price["High"],
                "direction": self.direction,
            }

    def bump_af(self):
        self.AF += self.AFstep
        if self.AF > self.AFmax:
            self.AF = self.AFmax

    def calc(self, this_price, history: List[Price]) -> Output:
        if len(history) <= self.start_period:
            if len(history) == self.start_period:
                return self.get_starting_PSAR(history, this_price)
            return None

        last_output: Output = history[-1][self.name]

        PSARextreme = last_output["PSARextreme"]
        PStARt = None

        if self.direction == "RISING":
            # calculate PSAR
            PSAR = last_output["PSAR"] + self.AF * (
                last_output["PSARextreme"] - last_output["PSAR"]
            )

            # if new extreme
            # save new extreme & increment AF
            if this_price["High"] > last_output["PSARextreme"]:
                PSARextreme = this_price["High"]
                self.bump_af()

            # check last two prices limiter
            if history[-1]["Low"] < PSAR:
                PSAR = history[-1]["Low"]
            elif history[-2]["Low"] < PSAR:
                PSAR = history[-2]["Low"]

            # if current psar within current price range,
            # flip direction, reset AF, & set psar = psar extreme
            if this_price["Low"] <= PSAR:
                self.AF = self.AFstart
                PStARt = PSAR
                PSAR = PSARextreme
                self.direction = "FALLING"
                PSARextreme = this_price["Low"]

        elif self.direction == "FALLING":
            # calculate PSAR
            PSAR = last_output["PSAR"] - self.AF * (
                last_output["PSAR"] - last_output["PSARextreme"]
            )

            # if new extreme
            # save new extreme & increment AF
            if this_price["Low"] < last_output["PSARextreme"]:
                PSARextreme = this_price["Low"]
                self.bump_af()

            # check last two prices limiter
            if history[-1]["High"] > PSAR:
                PSAR = history[-1]["High"]
            elif history[-2]["High"] > PSAR:
                PSAR = history[-2]["High"]

            # if current psar within current price range,
            # flip direction, reset AF, & set psar = psar extreme
            if PSAR <= this_price["High"]:
                self.AF = self.AFstart
                PStARt = PSAR
                PSAR = PSARextreme
                self.direction = "RISING"
                PSARextreme = this_price["High"]

        else:
            raise Error("Invalid PSAR direction: " + self.direction)

        return {
            "PSAR": PSAR,
            "PSARextreme": PSARextreme,
            "direction": self.direction,
            "PStARt": PStARt,
        }
