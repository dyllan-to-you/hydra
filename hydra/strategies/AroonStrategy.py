from typing import Tuple
from numpy import NaN
from . import Strategy
from hydra.indicators import Indicator, Decision, Aroon
import pandas as pd
from bokeh.plotting import figure


class AroonStrategy(Strategy):
    indicator: Indicator

    def __init__(self, indicator=Aroon.Indicator, period=25):
        super().__init__()
        self.period = period
        self.indicator = indicator(period)
        self.name = f"AroonStrategy({self.indicator.name})"

    def init_indicators(self):
        return self.indicator

    def _decide(self, price_history) -> Tuple[Decision, float]:

        if len(price_history) <= self.period + 1:
            return (Decision.NONE, NaN)

        try:
            this_price = price_history[-1]
            this_indicator = this_price[self.indicator.name]
            last_indicator = price_history[-2][self.indicator.name]
            if (
                last_indicator["oscillator"] < 100
                and this_indicator["oscillator"] >= 100
            ):
                return self.pick_price(Decision.BUY, this_price)

            if (
                last_indicator["oscillator"] > -100
                and this_indicator["oscillator"] <= -100
            ):
                return self.pick_price(Decision.SELL, this_price)
        except IndexError:
            pass

        return (Decision.NONE, NaN)

    def pick_price(self, decision, price) -> Tuple[Decision, float]:
        if decision == Decision.BUY:
            if price[self.indicator.name]["up"] < 100:
                return decision, price["Open"]
            return decision, price[self.indicator.name]["peak"]

        if decision == Decision.SELL:
            if price[self.indicator.name]["down"] < 100:
                return decision, price["Open"]
            return decision, price[self.indicator.name]["valley"]

    def draw_bokeh_subplot(self, plot, ctx):
        p = figure(
            plot_height=250,
            plot_width=1200,
            x_range=plot.x_range,
            x_axis_type="datetime",
        )
        p.line(
            x="Date", y=f"{self.indicator.name}.down", color="red", source=ctx.history
        )
        p.line(
            x="Date", y=f"{self.indicator.name}.up", color="green", source=ctx.history
        )
        return p