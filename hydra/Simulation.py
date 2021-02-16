import time
from datetime import datetime
from hydra.strategies import Decision, DecisionEvent
import os
import numpy as np
import pandas as pd
from csv import DictReader
from typing import List, NamedTuple, TypedDict, cast
import dateutil.parser as parser
from hydra import Hydra
from hydra.types import Price
from hydra.strategies.AroonStrategy import AroonStrategy
from hydra.strategies.AroonOpenStrategy import AroonOpenStrategy
from hydra.indicators import Aroon, AroonTulip
from tqdm import tqdm
import asyncio
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Order(NamedTuple):
    time: datetime
    price: float


class Trade(TypedDict, total=False):
    buy: Order
    sell: Order
    profit: float
    pl: float
    up: float
    down: float


class Simulation:
    cash: int
    trade_history: List[Trade]
    hydra: Hydra

    def __init__(self, hydra, cash=1):
        self.hydra = hydra
        self.trade_history = []
        self.cash = 1

    def tick(self, price: Price):
        (decision, decision_price), indicated_price = self.hydra.feed(price)
        fee = 1 - (self.hydra.fee / 100)
        order: Order = Order(price["Date"], decision_price)

        if decision == Decision.BUY:
            trade: Trade = {
                "buy": order,
                "profit": fee,
            }
            self.cash *= fee
            self.trade_history.append(trade)
        elif decision == Decision.SELL:
            trade = self.trade_history[-1]
            buy_time, buy_price = trade["buy"]
            sell_time, sell_price = trade["sell"] = order
            trade["pl"] = sell_price / buy_price
            self.cash *= trade["pl"] * fee
            trade["profit"] *= trade["pl"] * fee


def run_sim(rangeStart=0, rangeEnd=2.5, periods=10, graph=False):
    result = []
    simulations = []
    for i in range(2, periods):
        period = i
        # simulations.append(
        #     Simulation(
        #         Hydra(
        #             AroonStrategy(AroonTulip.Indicator, period),
        #             name="Tulip",
        #             fee=0.06,
        #         )
        #     )
        # )
        simulations.append(
            Simulation(
                Hydra(
                    AroonStrategy(Aroon.Indicator, period),
                    name="DIY",
                    fee=0.06,
                )
            ),
        )

    with pd.option_context("display.max_rows", None, "display.max_columns", 0):
        result = []
        with open(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "../data/Bitfinex_BTCUSD_1h.test.csv",
            )
        ) as file:
            prices = DictReader(file)
            for idx, row in tqdm(enumerate(prices)):
                if idx < rangeStart * 8760 or idx >= rangeEnd * 8760:
                    continue

                price: Price = {
                    "Date": cast(str, row.get("date")),
                    "Open": float(cast(str, row.get("open"))),
                    "High": float(cast(str, row.get("high"))),
                    "Low": float(cast(str, row.get("low"))),
                    "Close": float(cast(str, row.get("close"))),
                    "Volume": float(cast(str, row.get("Volume BTC"))),
                    "Volume_USD": float(cast(str, row.get("Volume USD"))),
                }
                for sim in simulations:
                    sim.tick(price)
            for sim in simulations:
                if graph:
                    df = sim.hydra.price_history_df
                    trades = pd.json_normalize(sim.trade_history, sep=".")
                    buys = pd.DataFrame(
                        trades["buy"].tolist(), columns=["Date", "Price"]
                    )
                    # buys.set_index("Date", inplace=True)
                    sells = pd.DataFrame(
                        trades["sell"].tolist(), columns=["Date", "Price"]
                    )
                    # sells.set_index("Date", inplace=True)

                    aup = f"{sim.hydra.strategy.indicator.name}.up"
                    adown = f"{sim.hydra.strategy.indicator.name}.down"
                    hovertext = []
                    for i in range(len(df["Open"])):
                        hovertext.append(
                            f"AroonUp: {str(df[aup][i])} <br>AroonDown: {str(df[adown][i])}"
                        )

                    fig = make_subplots(
                        rows=2, cols=1, start_cell="top-left", shared_xaxes=True
                    )
                    fig.add_trace(
                        go.Candlestick(
                            x=df.index,
                            open=df["Open"],
                            high=df["High"],
                            low=df["Low"],
                            close=df["Close"],
                            customdata=df[
                                [
                                    f"{sim.hydra.strategy.indicator.name}.up",
                                    f"{sim.hydra.strategy.indicator.name}.down",
                                ]
                            ],
                            text=hovertext,
                            hoverinfo="all",
                        ),
                        row=2,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            name="Buys",
                            marker_color="green",
                            x=buys["Date"],
                            y=buys["Price"],
                            text=buys["Price"],
                            mode="markers",
                            marker_line_width=2,
                            marker_size=10,
                        ),
                        row=2,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            name="Sells",
                            marker_color="red",
                            x=sells["Date"],
                            y=sells["Price"],
                            text=sells["Price"],
                            mode="markers",
                            marker_line_width=2,
                            marker_size=10,
                        ),
                        row=2,
                        col=1,
                    )

                    fig.add_trace(
                        go.Scatter(
                            name="Aroon Up",
                            marker_color="green",
                            x=df.index,
                            y=df[f"{sim.hydra.strategy.indicator.name}.up"],
                            mode="lines",
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            name="Aroon Down",
                            marker_color="red",
                            x=df.index,
                            y=df[f"{sim.hydra.strategy.indicator.name}.down"],
                            mode="lines",
                        ),
                        row=1,
                        col=1,
                    )
                    fig.update_layout(
                        title=sim.hydra.name, xaxis2_rangeslider_visible=False
                    )

                    fig.show()
                
                # print(pd.DataFrame(sim.trade_history))
                result.append(
                    {
                        "name": sim.hydra.name,
                        "strategy": sim.hydra.strategy.name,
                        "Total": sim.cash,
                        "Transactions": len(sim.trade_history),
                    }
                )
        df = pd.DataFrame(result)
        print(df)


# steps = 1 / 12
# for loop in np.arange(0, 3, steps):
#     run_sim(loop, (loop + steps))
t0 = time.time()
# for loop in np.arange(2, 50, 1):
run_sim(0, 3, 100)
t1 = time.time()
print("Total Time Elapsed:", t1-t0)
