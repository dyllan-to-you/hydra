from typing import Tuple
from . import Strategy
from hydra.indicators import Aroon, Indicator, Decision, PSAR
from bokeh.plotting import figure


class PSARoonStrategy(Strategy):
    indicator: Indicator

    def __init__(
        self,
        psar_indicator=PSAR.Indicator,
        aroon_indicator=Aroon.Indicator,
        aroon_buy_threshold=25,
        **kwargs,
    ):
        super().__init__()
        self.aroon_buy_threshold = aroon_buy_threshold
        self.args = kwargs
        self.psar_indicator = psar_indicator(**kwargs)
        self.aroon_indicator = aroon_indicator(**kwargs)
        self.shortname = "PSARoonStrategy"
        self.name = (
            f"{self.shortname}({self.psar_indicator.name}, {self.aroon_indicator.name})"
        )

    def init_indicators(self):
        return [
            self.psar_indicator,
            self.aroon_indicator,
        ]

    def _decide(self, price_history) -> Tuple[Decision, float]:

        # start period represents the range of numbers to skip
        # and choose the first extreme.
        # We multiply by 5 to ensure we don't get any false
        # buys/sells while the PSAR is calibrating
        price_history_len = len(price_history)
        # print(
        #     price_history_len,
        #     self.psar_indicator.start_period,
        #     self.aroon_indicator.period,
        # )
        if (
            price_history_len <= self.psar_indicator.start_period * 5
            or price_history_len <= self.aroon_indicator.period + 1
        ):
            # print(price_history)
            return (Decision.NONE, None)

        try:
            this_price = price_history[-1]
            last_price = price_history[-2]
            # print(this_price)
            this_psar = this_price[self.psar_indicator.name]
            last_psar = last_price[self.psar_indicator.name]

            this_aroon = this_price[self.aroon_indicator.name]
            # last_aroon = last_price[self.aroon_indicator.name]
            if (
                this_psar["direction"] != last_psar["direction"]
                and this_psar["direction"] == "RISING"
                and this_aroon["oscillator"] > self.aroon_buy_threshold
            ):
                return self.pick_price(Decision.BUY, this_price)

            if (
                this_psar["direction"] != last_psar["direction"]
                and this_psar["direction"] == "FALLING"
            ):
                decision = Decision.SELL
                return (decision, this_psar["PStARt"])

        except IndexError:
            pass
        # except KeyError:
        #     pass

        return Decision.NONE, None

    def pick_price(self, decision, price) -> Tuple[Decision, float]:
        if decision == Decision.BUY:
            if price[self.aroon_indicator.name]["up"] < 100:
                return decision, price["Open"]
            return decision, price[self.aroon_indicator.name]["peak"]

        if decision == Decision.SELL:
            if price[self.aroon_indicator.name]["down"] < 100:
                return decision, price["Open"]
            return decision, price[self.aroon_indicator.name]["valley"]

    def draw_bokeh_layer(self, plot, ctx):
        plot.circle(
            x="Date",
            y=f"{self.psar_indicator.name}.PSAR",
            size=5,
            color="black",
            alpha=1,
            source=ctx.history,
        )

    def draw_bokeh_subplot(self, plot, ctx):
        p = figure(
            plot_height=250,
            plot_width=1200,
            x_range=plot.x_range,
            x_axis_type="datetime",
        )
        p.line(
            x="Date",
            y=f"{self.aroon_indicator.name}.down",
            color="red",
            source=ctx.history,
        )
        p.line(
            x="Date",
            y=f"{self.aroon_indicator.name}.up",
            color="green",
            source=ctx.history,
        )
        return p