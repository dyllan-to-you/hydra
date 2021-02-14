from datetime import datetime
from hydra.strategies import Decision, DecisionEvent
import os
import numpy as np
import pandas as pd
from csv import DictReader
from typing import List, NamedTuple, TypedDict, cast
from unittest.mock import patch
import dateutil.parser as parser
from hydra import Hydra
from hydra.types import Price
from hydra.strategies.AroonStrategy import AroonStrategy
from hydra.indicators import Aroon, AroonTulip
from tqdm import tqdm
import asyncio
import random


def pick_price_avg(row, decision, **a):
    return (row["Open"] + row["Close"]) / 2


def pick_price_rand(row, decision, **a):
    return random.uniform(row["Open"], row["Close"])


def pick_price_last_peak(row, decision, name):
    if decision == Decision.BUY:
        if row[name]["up"] < 100:
            return row["Open"]
        return row[name]["peak"]

    if decision == Decision.SELL:
        if row[name]["down"] < 100:
            return row["Open"]
        return row[name]["valley"]


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

    def __init__(self, hydra, cash=1):
        self.hydra = hydra
        self.trade_history = []
        self.cash = 1

    def tick(self, price: Price, pick_price=pick_price_last_peak):
        decision, indicated_price = self.hydra.feed(price)
        fee = 1 - (self.hydra.fee / 100)
        name = self.hydra.strategy.indicator.name
        order: Order = Order(
            price["Date"],
            pick_price(indicated_price, decision, name=name),
        )

        if decision == Decision.BUY:
            trade: Trade = {
                "buy": order,
                "profit": fee,
                "up": indicated_price[name]["up"],
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
            trade["down"] = indicated_price[name]["down"]


def run_sim(rangeStart=0, rangeEnd=2.5):
    result = []
    simulations = [
        Simulation(
            Hydra(
                AroonStrategy(Aroon.Indicator, 16),
                name="DIY",
                fee=0.06,
            )
        ),
        Simulation(
            Hydra(
                AroonStrategy(AroonTulip.Indicator, 16),
                name="Tulip",
                fee=0.06,
            )
        ),
    ]
    # for i in range(1, 25):
    #     period = i * 2
    #     simulations.append(
    #         Simulation(
    #             Hydra(
    #                 AroonStrategy(AroonTulip.Indicator, period),
    #                 name="binance",
    #                 fee=0.06,
    #             )
    #         )
    #     )

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
                    "Date": parser.parse(cast(str, row.get("date"))),
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
                print(
                    sim.hydra.name,
                    sim.hydra.strategy.name,
                    sim.cash,
                    len(sim.trade_history),
                )
                print(
                    pd.json_normalize(sim.trade_history, sep=".").drop(columns="profit")
                )
        #         result.append(
        #             {
        #                 "name": sim.hydra.name,
        #                 "strategy": sim.hydra.strategy.name,
        #                 "Total": sim.cash,
        #                 "Transactions": len(sim.trade_history),
        #             }
        #         )
        # df = pd.DataFrame(result)
        # print(df)


# steps = 1 / 12
# for loop in np.arange(0, 3, steps):
#     run_sim(loop, (loop + steps))
run_sim(0, 1)
