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
import pyarrow.parquet as pq
import pathlib
from datetime import datetime
import hydra.PlotlyChart as pl
import hydra.BokehChart as bokeh


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
            trade["cash_buy"] = self.cash
            self.trade_history.append(trade)
        elif decision == Decision.SELL:
            trade = self.trade_history[-1]
            buy_time, buy_price = trade["buy"]
            sell_time, sell_price = trade["sell"] = order
            trade["pl"] = sell_price / buy_price
            self.cash *= trade["pl"] * fee
            trade["cash_sell"] = self.cash
            trade["profit"] *= trade["pl"] * fee


def run_sim(
    pair,
    startDate="2020-01-01",
    endDate="2021-01-01",
    periods=[10],
    interval=1,
    graph=False,
    path="./data/kraken/",
):
    result = []
    simulations = []

    for period in periods:
        simulations.append(
            Simulation(
                Hydra(
                    AroonStrategy(Aroon.Indicator, period),
                    name="DIY",
                    fee=0.06,
                )
            ),
        )

    result = []
    filepath = pathlib.Path().absolute()
    parqpath = filepath.joinpath(path, "parq", f"{pair}_{interval}.parq")
    table = pq.read_table(parqpath)
    prices = table.to_pandas()
    prices["time"] = pd.to_datetime(prices["time"], unit="s")
    prices = prices.set_index("time").loc[startDate:endDate]
    # prices = prices.append(pd.DataFrame([{"trades": 0}]))
    # print(prices["trades"].value_counts(dropna=False).sort_index())
    # return

    last_row = None
    total_intervals = (
        pd.Timedelta(prices.index[0] - prices.index[-1]).total_seconds()
        / 60.0
        / interval
    )
    increment = 1 / len(simulations)
    with tqdm(total=total_intervals, unit_scale=True, unit="intervals") as pbar:
        for row in prices.itertuples(index=True):
            time, open, high, low, close, volume, trades = row

            price: Price = {
                "Date": time,
                "Open": open,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
                "Trades": trades,
            }

            # last row is none for first value
            # So don't fill gaps if last_row is none
            # Otherwise, fill gaps if the gap between times is greater than the interval
            gaps = []
            while (
                last_row is not None
                and pd.Timedelta(time - last_row[0]).total_seconds() / 60.0 > interval
            ):
                new_time = last_row[0] + pd.Timedelta(minutes=interval)
                last_row = (new_time, close, close, close, close, 0, 0)
                fill_price: Price = {
                    "Date": new_time,
                    "Open": close,
                    "High": close,
                    "Low": close,
                    "Close": close,
                    "Volume": 0,
                    "Trades": 0,
                }
                for sim in simulations:
                    pbar.update(increment)
                    sim.tick(fill_price)
                if len(gaps) == 0 or len(gaps) == 1:
                    gaps.append(new_time)
                else:
                    gaps[1] = new_time

            # if len(gaps):
            #     print("Filled Gap:", gaps)
            last_row = row

            for sim in simulations:
                pbar.update(increment)
                sim.tick(price)

    for sim in simulations:
        if graph:
            chart = bokeh.Chart(interval=interval, bars_to_display=60 * 24 * 30)
            chart.graph(sim)

            # Render full year's png
            # imagedir = filepath.joinpath("images", pair)
            # os.makedirs(imagedir, exist_ok=True)
            # fig = pl.graph(sim)
            # name = f"{sim.hydra.name}.{sim.hydra.strategy.name}.all.png"
            # print("Writing", imagedir, name)
            # fig.write_image(imagedir.joinpath(name))
            # os.system("start " + imagedir.joinpath(name))

            # current_time = df.index[0]
            # while current_time < df.index[-1]:
            #     next_time = current_time + pd.Timedelta(months=1)
            #     month_fig = pl.graph(sim, df.iloc[current_time:next_time])
            #     name = f"{sim.hydra.name}.{sim.hydra.strategy.name}.{current_time.year}.{current_time.month}.html"
            #     print("Writing", name)
            #     month_fig.write_html(imagedir.joinpath(name))
            #     current_time = next_time

        # print(pd.DataFrame(sim.trade_history))
        result.append(
            {
                "name": sim.hydra.name,
                "strategy": sim.hydra.strategy.name,
                "Total": sim.cash,
                "Transactions": len(sim.trade_history),
            }
        )
    with pd.option_context("display.max_rows", None, "display.max_columns", 0):
        df = pd.DataFrame(result)
        print(df)
    return simulations


periods = [35]
# aroon(40  hours) = aroon(2400 min)
# for i in range(1, 160):
#     periods.append(i * 15)

t0 = time.time()
run_sim(
    "XBTUSD",
    startDate="2018-05-15",
    endDate="2018-07-15",
    interval=1,
    periods=[30],
    graph=True,
)
# run_sim(
#     "XBTUSD",
#     startDate="2018-05-15",
#     endDate="2021-05-15",
#     interval=60,
#     periods=[35],
#     graph=True,
# )
t1 = time.time()
print("Total Time Elapsed:", t1 - t0)
