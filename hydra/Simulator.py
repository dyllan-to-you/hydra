import collections
import itertools
import sys
import traceback
import psutil
import os
import json
import pathlib
import platform
import pprint
from datetime import datetime, timedelta
from operator import itemgetter
from typing import Dict, List, Set, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
from numba import njit
from pyarrow.types import is_list
import ray
from ray.util.queue import Queue
from sqlalchemy.orm.instrumentation import instance_state
from tqdm import tqdm

from hydra.models import BuyOrder, Direction, SellOrder
from hydra.utils import chunk_list, now, printd, timeme
from hydra.money import calculate_profit, get_decay, get_decay_profit
from hydra.DataLoader import load_parquet_by_date, ParquetLoader

pp = pprint.PrettyPrinter(indent=2)
project_dir = pathlib.Path(__file__).absolute().parent.parent
NUM_CORES = psutil.cpu_count(logical=True)

pair = "XBTUSD"
startDate = pd.to_datetime("2017-05-01")
endDate = pd.to_datetime("2020-01-01")
aroon_reference = "2021-05-20T1919"
order_folder = "order 2021-06-16T1632 fee=0.001"
output_dir = project_dir / "output" / pair
order_dir = output_dir / "orders" / order_folder

fee = 0.001
init_minutes = 20160

# PARAMS ###############################
window_size = 500
decay_rate = 0.05 ** (1 / 20160)

min_trade_profit = 1.003
min_profit = 1.01
min_success_ratio = 0.8
min_window_trades = 3
margin = 0.99

# #######################################
# window_size = dict(start=300, end=1 * 30 * 24 * 60, step_percentage=25)
# min_trade_profit = dict(start=1.001, end=1.03, step=0.001)
# min_profit = 1.03
# min_success_ratio = 0.8
# min_window_trades = 5
# margin = 0.99
# decay_rate = 0.05 ** (1 / 40320)
# ------------------------------------


buy_fee = 1 + fee
sell_fee = 1 - fee


def datetime_range(start, end, delta, inclusive=False):
    current = start
    while current < end:
        yield current
        current += delta
    if inclusive:
        yield current


class Portfolio:
    just_ordered: bool
    orders: List[Union[BuyOrder, SellOrder]]
    direction: Direction
    total_profit: float
    fee_loss: float

    def __init__(self):
        self.orders = []
        self.direction = Direction.BUY
        self.total_profit = 1.0
        self.fee_loss = 1.0

    def find_order(self, best_simulations, current_order_df):
        if current_order_df is None:
            return None

        if isinstance(best_simulations, list):
            best_orders = current_order_df.loc[
                current_order_df["simulation_id"].isin(best_simulations)
            ]
        elif isinstance(best_simulations, np.int64):
            best_orders = current_order_df.loc[
                current_order_df["simulation_id"] == best_simulations
            ]
        else:
            print(f"unexpected, best_orders is instance of {type(best_simulations)}")

        return best_orders.iloc[0] if not best_orders.empty else None

    def add_buy_order(self, best_simulations, current_order_df, pbar):
        order = self.find_order(best_simulations, current_order_df)

        if order is None:
            return None
        [timestamp, simulation_id, trigger_price] = order
        order = BuyOrder(timestamp, trigger_price, simulation_id)
        pbar.write(f"[{now()}] Buying {order}")
        self.orders.append(order)
        self.direction = Direction.SELL
        self.just_ordered = True
        return simulation_id

    def add_sell_order(self, best_simulations, current_order_df, pbar):
        order = self.find_order(best_simulations, current_order_df)

        if order is None:
            return None

        [timestamp, simulation_id, trigger_price, profit, cycle_time, hold_time] = order
        last_order: BuyOrder = self.orders[-1]
        profit = calculate_profit(last_order[1], trigger_price, buy_fee, sell_fee)
        order = SellOrder(timestamp, trigger_price, profit)
        self.total_profit *= profit
        self.fee_loss *= sell_fee / buy_fee
        pbar.write(f"[{now()}] Selling {order},{self.total_profit=},{self.fee_loss=}")
        self.orders.append(order)
        self.direction = Direction.BUY
        self.just_ordered = True


class Simulation:
    id: int
    decays: npt.ArrayLike
    last_decay: int
    cumm_hold_time: int
    success_total: int
    window_history: int
    profit_total: float

    def __init__(self, id, last_decay=0):
        self.id = id
        self.last_decay = last_decay
        self.decay = 1
        self.cumm_hold_time = 0
        self.success_total = 0
        self.window_history = 0
        self.profit_total = 1

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, o: object) -> bool:
        return self.id == o.id

    def add_profit(self, profit, current_time, hold_time):
        self.cumm_hold_time += hold_time
        elapsed = current_time - self.last_decay
        if profit >= min_trade_profit:
            self.success_total += 1
        self.window_history += 1

        self.profit_total *= profit

        self.decay = get_decay_profit(
            self.decay,
            decay_rate,
            profit,
            elapsed,
        )
        self.last_decay = current_time

    def remove_profit(self, profit):
        if profit >= min_trade_profit:
            self.success_total -= 1

        self.window_history -= 1

        self.profit_total /= profit


@ray.remote
class SimulationActor:
    tracked_chunks: List[int]
    history: Queue
    simulations: Dict[int, Simulation]
    threshold_simulations: Set[Simulation]
    window_size: int

    def __init__(self, tracked_chunks, window_size=window_size):
        self.tracked_chunks = tracked_chunks
        self.history = None
        self.simulations = dict()
        self.threshold_simulations = set()
        self.window_size = window_size

    def set_history(self, queue):
        self.history = queue
        return self.history

    def add_profits(self, sells, current_time_min):
        for sell in sells:
            (
                index,
                timestamp,
                simulation_id,
                trigger_price,
                profit,
                cycle_time,
                hold_time,
            ) = sell
            sim = self.simulations.setdefault(simulation_id, Simulation(simulation_id))
            sim.add_profit(profit, current_time_min, hold_time)
            if (
                sim.window_history > min_window_trades
                and sim.success_total / sim.window_history >= min_success_ratio
                and sim.profit_total >= min_profit
            ):
                self.threshold_simulations.add(sim)
            else:
                self.threshold_simulations.discard(sim)

    def remove_profits(self, sells):
        for sell in sells:
            (
                index,
                timestamp,
                simulation_id,
                trigger_price,
                profit,
                cycle_time,
                hold_time,
            ) = sell
            sim = self.simulations.get(simulation_id)

            if sim is not None:
                sim.remove_profit(profit)

                if (
                    sim.window_history > min_window_trades
                    and sim.success_total / sim.window_history >= min_success_ratio
                    and sim.profit_total >= min_profit
                ):
                    self.threshold_simulations.add(sim)
                else:
                    self.threshold_simulations.discard(sim)

    def sell_loader(self):
        for current_day in datetime_range(startDate, endDate, pd.Timedelta(1, "day")):
            try:
                sell_day = load_parquet_by_date(
                    order_dir, current_day, "sell", self.tracked_chunks
                )
            except Exception as e:
                printd(f"Couldn't find/load {e}")
                raise e
            # sell_day.drop("trigger_price", axis=1, inplace=True)
            sells = sell_day.groupby("timestamp")
            for sell in sells:
                yield sell

    def run_simulation(self):
        counter = 0
        old_sell_date = None
        main_sells = self.sell_loader().__iter__()
        window_sells = self.sell_loader().__iter__()

        for current_day in datetime_range(startDate, endDate, pd.Timedelta(1, "day")):
            current_sell_date, current_sell_df = next(main_sells, (None, None))
            history = collections.deque()
            for current_minute in datetime_range(
                current_day,
                current_day + pd.Timedelta(1, "day"),
                pd.Timedelta(1, "minute"),
            ):
                counter += 1
                if current_sell_date is None:
                    history.append((current_minute, []))
                elif current_sell_date == current_minute:
                    if counter > self.window_size:
                        expected_window_start = current_sell_date - timedelta(
                            minutes=self.window_size
                        )
                        while (
                            old_sell_date is None
                            or old_sell_date < expected_window_start
                        ):
                            old_sell_date, old_sell_df = next(
                                window_sells, (None, None)
                            )
                            self.remove_profits(old_sell_df.itertuples(name=None))

                    self.add_profits(
                        current_sell_df.itertuples(name=None),
                        counter,
                    )

                    if counter > self.window_size:
                        best_sim = self.get_best_simulations(
                            self.threshold_simulations,
                        )
                        history.append((current_minute, best_sim))
                    else:
                        history.append((current_minute, []))

                    current_sell_date, current_sell_df = next(main_sells, (None, None))
                else:
                    history.append((current_minute, []))
            if len(history) == 0:
                print(self.tracked_chunks, history)
            self.history.put((current_day, history))

    def get_best_simulations(
        self,
        simulations,
    ):
        profits = [
            (
                sim.id,
                sim.decay * 0.25 + weighted_profit,
                weighted_profit,
                sim.success_total,
                sim.profit_total,
                sim.cumm_hold_time,
            )
            # TODO: [df] loop through df
            for sim in simulations
            if (
                weighted_profit := (
                    (sim.profit_total - 1) * (sim.success_total / sim.window_history)
                )
                + 1
            )
            >= min_profit
        ]
        if not len(profits):
            return []

        best_profit = max(profits, key=itemgetter(1))[1]
        minimum = best_profit * margin
        best = [
            (
                simId,
                decay_avg,
                weighted_profit,
                success_count,
                profit_total,
                cumm_hold_time,
            )
            for simId, decay_avg, weighted_profit, success_count, profit_total, cumm_hold_time, *_ in profits
            if decay_avg > minimum
        ]

        return best


@timeme
def main(pair, startDate, endDate, aroon_reference):

    printd("Setting the stage")
    cores = [
        int(f.name.split("=")[1])
        for f in os.scandir(
            order_dir / f"year={startDate.year}/month={startDate.month}"
        )
        if f.is_dir() and f.name.startswith("core")
    ]

    core_chunks = chunk_list(cores, NUM_CORES)
    actors = {}
    for chunk in core_chunks:
        actor = SimulationActor.remote(chunk)
        actors[actor] = {
            "history": ray.get(actor.set_history.remote(Queue(maxsize=30))),
            "chunks": chunk,
        }

    printd("Preparing to simp-de-sim and pull")
    total_ticks = pd.to_datetime(endDate) - pd.to_datetime(startDate)
    total_ticks = total_ticks.total_seconds() / 60
    with tqdm(total=total_ticks, unit="tick", smoothing=0) as pbar:
        counter = 0
        portfolio = Portfolio()
        buy_loader = ParquetLoader(order_dir, "buy")
        sell_loader = ParquetLoader(order_dir, "sell")
        best_simulations = []
        sim_actor_map = {}
        for actor in actors.keys():
            actor.run_simulation.remote()

        for current_day in datetime_range(startDate, endDate, pd.Timedelta(1, "day")):
            printd(f"{current_day} Loading Day", pbar=pbar)
            daily_tick = {
                actor: actor_props["history"].get()
                for actor, actor_props in actors.items()
            }
            printd(f"{current_day} Loaded Day", pbar=pbar)
            for actor, (tick_day, daily_history) in daily_tick.items():
                assert current_day == tick_day
            for current_minute in datetime_range(
                current_day,
                current_day + pd.Timedelta(1, "day"),
                pd.Timedelta(1, "minute"),
            ):
                pbar.update(1)
                counter += 1
                portfolio.just_ordered = False
                tick = {
                    actor: daily_history.popleft()
                    for actor, (tick_day, daily_history) in daily_tick.items()
                }
                # for actor, (tick_time, best_sim) in tick.items():
                #     assert current_minute == tick_time

                if portfolio.direction == Direction.SELL:
                    ideal_actor_chunks = actors[ideal_actor]["chunks"]
                    sells = sell_loader.get_file(current_minute, ideal_actor_chunks)
                    portfolio.add_sell_order(ideal_sim, sells, pbar=pbar)

                    if portfolio.just_ordered:
                        best_simulations = get_best_simulations(tick, sim_actor_map)

                if not portfolio.just_ordered and portfolio.direction == Direction.BUY:
                    if counter > init_minutes:
                        best_simulations = get_best_simulations(tick, sim_actor_map)

                    if len(best_simulations) > 0:
                        buys = buy_loader.get_file(current_minute, list(range(0, 64)))
                        ideal_sim = portfolio.add_buy_order(
                            best_simulations, buys, pbar=pbar
                        )
                        ideal_actor = sim_actor_map.get(ideal_sim, None)

        print(portfolio.orders)
        print(
            json.dumps(
                portfolio.orders,
                indent=2,
                default=lambda o: o.serialize() if hasattr(o, "serialize") else str(o),
            )
        )
        print(portfolio.total_profit)


# @timeme
def get_best_simulations(tick, sim_actor_map):
    profits = []

    for a, t in tick.items():
        # put best simulations for each actor into a single array
        profits.extend(t[1])

        # Map sim ID to actor
        for best_sim_candidate in t[1]:
            sim_actor_map[best_sim_candidate[0]] = a

    # get absolute best simulations
    if len(profits) == 0:
        return []
    # print(profits)
    best_profit = max(profits, key=itemgetter(1))[1]
    minimum = best_profit * margin
    best = [
        (simId, weighted_profit, decay_avg)
        for simId, decay_avg, weighted_profit, success_count, *_ in profits
        if decay_avg > minimum
    ]
    print(f"{len(best)=}")

    best_simulations = sorted(
        profits,
        key=itemgetter(2, 3),
        reverse=True,
    )

    # extract simulation id
    best_simulations = [candidate[0] for candidate in best_simulations]
    return best_simulations


if __name__ == "__main__":
    try:
        ray.init(include_dashboard=False, local_mode=False)
        # cProfile.run("main()")
        main(pair, startDate, endDate, aroon_reference)
    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print(e)
    finally:
        ray.shutdown()
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
