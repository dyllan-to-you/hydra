import math
import sys
import time
from datetime import datetime
from hydra.strategies import Decision, DecisionEvent
import numpy as np
import pandas as pd
from typing import List, NamedTuple, TypedDict, cast
import dateutil.parser as parser
from hydra import Hydra
from hydra.types import Price
from hydra.strategies.PSARoonStrategy import PSARoonStrategy
from hydra.strategies.AroonStrategy import AroonStrategy
from tqdm import tqdm
import pyarrow.parquet as pq
import pathlib
from datetime import datetime
import hydra.BokehChart as bokeh
import sqlite3
from datetime import datetime


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
        self.fee = 1 - (self.hydra.fee / 100)

    def tick(self, price: Price):
        (decision, decision_price), indicated_price = self.hydra.feed(price)
        if decision == Decision.NONE:
            return

        if decision == Decision.BUY:
            order: Order = Order(price["Date"], decision_price)
            trade: Trade = {
                "buy": order,
                "profit": self.fee,
            }
            self.cash *= self.fee
            trade["cash_buy"] = self.cash
            self.trade_history.append(trade)
        elif decision == Decision.SELL:
            order: Order = Order(price["Date"], decision_price)
            trade = self.trade_history[-1]
            buy_time, buy_price = trade["buy"]
            sell_time, sell_price = trade["sell"] = order
            trade["pl"] = sell_price / buy_price
            self.cash *= trade["pl"] * self.fee
            trade["cash_sell"] = self.cash
            trade["profit"] *= trade["pl"] * self.fee


def run_sim(
    pair,
    startDate="2020-01-01",
    endDate="2021-01-01",
    simulations=[],
    interval=1,
    graph=False,
    path="./data/kraken/",
):
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
            if last_row is not None:
                numGaps = (
                    pd.Timedelta(time - last_row[0]).total_seconds() / 60.0 / interval
                )
                if numGaps > 1:
                    for gap in range(math.ceil(numGaps)):
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

        # print(pd.DataFrame(sim.trade_history))
        result.append(
            {
                "name": sim.hydra.name,
                "strat": sim.hydra.strategy.shortname,
                "strategy": sim.hydra.strategy.name,
                **sim.hydra.strategy.args,
                "Total": sim.cash,
                "Transactions": len(sim.trade_history),
            }
        )
    return simulations, result


# trade history
# result + strategy parameters


simulations = []
# for i in range(600):
#     period = (i + 1) * 5
#     for j in range(5):
#         AFstart = (j + 1) * 0.01
#         for k in range(20):
#             AFstep = (k + 1) * 0.005
#             for l in range(21):
#                 AFmax = AFstart + (AFstep * l)
#                 for m in range(11):
#                     threshold = m * 10
#                     simulations.append(
#                         Simulation(
#                             Hydra(
#                                 PSARoonStrategy(
#                                     period=period,
#                                     AFstart=AFstart,
#                                     AFstep=AFstep,
#                                     AFmax=AFmax,
#                                     aroon_buy_threshold=threshold,
#                                 ),
#                                 name="DIY",
#                                 fee=0.06,
#                             )
#                         ),
#                     )

# for i in range(2):
#     period = (i + 1) * 5
#     for j in range(2):
#         AFstart = (j + 1) * 0.01
#         for k in range(2):
#             AFstep = (k + 1) * 0.005
#             for l in range(2):
#                 AFmax = AFstart + (AFstep * l)
#                 for m in range(2):
#                     threshold = m * 10
#                     simulations.append(
#                         Simulation(
#                             Hydra(
#                                 PSARoonStrategy(
#                                     period=period,
#                                     AFstart=AFstart,
#                                     AFstep=AFstep,
#                                     AFmax=AFmax,
#                                     aroon_buy_threshold=threshold,
#                                 ),
#                                 name="DIY",
#                                 fee=0.06,
#                             )
#                         ),
#                     )

for period in range(2, 10):
    simulations.append(
        Simulation(
            Hydra(
                AroonStrategy(
                    period=period,
                ),
                name="DIY",
                fee=0.06,
            )
        ),
    )


t0 = time.time()
sims, result = run_sim(
    "XBTUSD",
    startDate="2018-05-15",
    endDate="2021-07-15",
    interval=1,
    simulations=simulations,
    graph=False,
)


t1 = time.time()
print("Total Time Elapsed:", t1 - t0)


# with pd.option_context("display.max_rows", None, "display.max_columns", 0):
df = pd.DataFrame(result)
try:
    outPath = sys.argv[1]
except:
    outPath = "./data/output"

maxProfit = df["Total"].max()
filedir = pathlib.Path.cwd().joinpath(outPath).resolve()
now = datetime.today().strftime("%Y-%m-%dT%H%M%S")
filename = f"{now} {len(simulations)} Simulations {round(maxProfit, 4)} Profit"
filepath = filedir.joinpath(filename)

# con_string = filedir.joinpath(f"{filepath}.db")
# print(con_string)
# filedir.mkdir(parents=True, exist_ok=True)
# # con_string.touch()
# conn = sqlite3.connect(con_string)
# cursor = conn.cursor()
# cursor.execute("SELECT SQLITE_VERSION()")
# data = cursor.fetchone()
# print("SQLite version:", data)

# df.to_sql(name="strategies", con=conn)
for sim in sims:
    trades = pd.json_normalize(sim.trade_history)
    trades["strategy"] = sim.hydra.strategy.name
    trades[["buy_time", "buy_price"]] = pd.DataFrame(
        trades["buy"].tolist(), index=trades.index
    )
    trades[["sell_time", "sell_price"]] = pd.DataFrame(
        trades["sell"].tolist(), index=trades.index
    )

    trades.drop(["buy", "sell"], axis=1, inplace=True)
    # print(trades)
    # trades.to_sql(name="trades", if_exists="append", con=conn)
