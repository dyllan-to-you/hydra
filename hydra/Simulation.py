from datetime import datetime
from hydra.strategies import Decision, DecisionEvent
import os
import pandas as pd
from csv import DictReader
from typing import List, NamedTuple, TypedDict, cast
from unittest.mock import patch
import dateutil.parser as parser
from hydra import Hydra
from hydra.types import Price
from hydra.strategies.AroonStrategy import AroonStrategy
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
        return row[name]["last_peak"]

    if decision == Decision.SELL:
        if row[name]["down"] < 100:
            return row["Open"]
        return row[name]["last_valley"]


class Order(NamedTuple):
    time: datetime
    price: float


class Trade(TypedDict, total=False):
    buy: Order
    sell: Order
    profit: float
    pl: float


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
        order: Order = Order(
            price["Date"],
            pick_price(
                indicated_price, decision, name=self.hydra.strategy.indicator.name
            ),
        )

        if decision == Decision.BUY:
            trade: Trade = {"buy": order, "profit": fee}
            self.cash *= fee
            self.trade_history.append(trade)
        elif decision == Decision.SELL:
            trade = self.trade_history[-1]
            buy_time, buy_price = trade["buy"]
            sell_time, sell_price = trade["sell"] = order
            trade["pl"] = sell_price / buy_price
            self.cash *= trade["pl"] * fee
            trade["profit"] *= trade["pl"] * fee

    # def run(
    #     self,
    #     year=None,
    # ):
    #     with open(
    #         os.path.join(
    #             os.path.dirname(os.path.realpath(__file__)),
    #             "../data/Bitfinex_BTCUSD_1h.test.csv",
    #         )
    #     ) as file:
    #         prices = DictReader(file)
    #         for idx, row in tqdm(enumerate(prices), desc="Processing Time"):
    #             if year is not None and (idx < year * 8760 or idx >= (year + 1) * 8760):
    #                 continue

    #             price: Price = {
    #                 "Date": parser.parse(cast(str, row.get("date"))),
    #                 "Open": float(cast(str, row.get("open"))),
    #                 "High": float(cast(str, row.get("high"))),
    #                 "Low": float(cast(str, row.get("low"))),
    #                 "Close": float(cast(str, row.get("close"))),
    #                 "Volume": float(cast(str, row.get("Volume BTC"))),
    #                 "Volume_USD": float(cast(str, row.get("Volume USD"))),
    #             }

    #             self.tick(price)

    # for hydra in self.hydras:
    #     decision_history = hydra.strategy.decision_history_df
    #     priced_decisions = hydra.price_history_df.join(
    #         decision_history, how="right"
    #     )

    #     buy_decision = None
    #     cash = 1
    #     history = []
    #     for index, row in priced_decisions.iterrows():
    #         if buy_decision is None:
    #             cash *= 1 - (hydra.fee / 100)
    #             buy_decision = row
    #             continue

    #         buy = pick_price(
    #             buy_decision, Decision.BUY, name=hydra.strategy.indicator.name
    #         )
    #         sell = pick_price(
    #             row, Decision.SELL, name=hydra.strategy.indicator.name
    #         )

    #         pl = sell / buy
    #         cash *= pl
    #         cash *= 1 - (hydra.fee / 100)

    #         history.append((pl, cash))

    #         buy_decision = None
    #         pass

    #     res.append(
    #         {
    #             "name": hydra.name,
    #             "strategy": hydra.strategy.name,
    #             "Total": history[-1][1],
    #             "Transactions": len(history),
    #         }
    #     )
    # return res


result = []
simulations = []
for i in range(1, 25):
    period = i * 2
    # Hydra(AroonStrategy(period), name="free trade"),
    # Hydra(AroonStrategy(period), name="kraken (taker)", fee=0.26),
    # Hydra(AroonStrategy(period), name="kraken (maker)", fee=0.16),
    simulations.append(
        Simulation(Hydra(AroonStrategy(period), name="binance", fee=0.06))
    )
    # [
    #     "binance(pro)",
    #     period,
    #     Simulation(Hydra(AroonStrategy(period)), fee=0.075),
    # ],
    # [
    #     "binance(pro + ref)",
    #     period,
    #     Simulation(Hydra(AroonStrategy(period)), fee=0.075 * 0.8),
    # ],

    # result.extend(sim.run(1, pick_price_last_peak))
    # result.extend(sim.run(2, pick_price_last_peak))
with pd.option_context("display.max_rows", None, "display.max_columns", None):
    result = []
    year = 2
    with open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../data/Bitfinex_BTCUSD_1h.test.csv",
        )
    ) as file:
        prices = DictReader(file)
        for idx, row in tqdm(enumerate(prices)):
            if year is not None and (idx < year * 8760 or idx >= (year + 1) * 8760):
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
