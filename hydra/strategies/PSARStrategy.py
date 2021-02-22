import pprint
from typing import Tuple
from numpy import NaN
from . import Strategy
from hydra.indicators import Indicator, Decision, PSAR
import pandas as pd

pp = pprint.PrettyPrinter(indent=4)


class PSARStrategy(Strategy):
    indicator: Indicator

    def __init__(self, indicator=PSAR.Indicator, **kwargs):
        super().__init__()
        self.args = kwargs
        self.indicator = indicator(**kwargs)
        self.name = f"PSARStrategy({self.indicator.name})"

    def init_indicators(self):
        return [
            self.indicator,
        ]

    def _decide(self, price_history) -> Tuple[Decision, float]:

        # start period represents the range of numbers to skip
        # and choose the first extreme.
        # We multiply by 5 to ensure we don't get any false
        # buys/sells while the PSAR is calibrating
        if len(price_history) <= self.indicator.start_period * 5:
            return (Decision.NONE, None)

        this_indicator = price_history[-1][self.indicator.name]
        # print("==========", this_indicator)
        # pp.pprint(price_history)

        if (
            this_indicator["direction"]
            != price_history[-2][self.indicator.name]["direction"]
        ):
            decision = (
                Decision.BUY
                if this_indicator["direction"] == "RISING"
                else Decision.SELL
            )
            return (decision, this_indicator["PStARt"])

        return (Decision.NONE, None)

    def draw_bokeh_layer(self, plot, ctx):
        plot.circle(
            x="Date",
            y=f"{self.indicator.name}.PSAR",
            size=5,
            color="black",
            alpha=1,
            source=ctx.history,
        )