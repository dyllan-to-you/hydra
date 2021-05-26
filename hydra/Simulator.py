import json
import pathlib
import platform
import pprint
from datetime import datetime
from operator import itemgetter
from typing import Dict, List, Set, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandas.core.groupby as pdGroupby
import pyarrow as pa
import pyarrow.parquet as pq
from numba import njit
from tqdm import tqdm

from hydra.models import BuyOrder, Direction, SellOrder
from hydra.utils import now, printd, timeme
from hydra.money import calculate_profit, get_decay

pp = pprint.PrettyPrinter(indent=2)


pair = "XBTUSD"
startDate = pd.to_datetime("2018-01-01")
endDate = pd.to_datetime("2019-01-01")
aroon_reference = "2021-05-20T1919"
order_folder = "orders 2021-05-22T0531 fee=0.001"

threshold_options = {"min_profit": 1.015, "min_trade_history": 20}
fee = 0.001
buy_fee = 1 + fee
sell_fee = 1 - fee


def load_output_signal(output_dir, file) -> pd.DataFrame:
    path = output_dir / f"{file}.parquet"
    return pq.read_table(path).to_pandas()


def load_parquet_by_date(dir, date, name):
    parquet_dir = f"year={date.year}/month={date.month}"
    sell_filename = f"day={date.day}.{name}"
    return load_output_signal(dir / parquet_dir, sell_filename)


def datetime_range(start, end, delta, inclusive=False):
    current = start
    while current < end:
        yield current
        current += delta
    if inclusive:
        yield current


decay_rates = np.array(
    [
        0.05 ** (1 / 180),
        0.05 ** (1 / 360),
        0.05 ** (1 / 525600),
    ]
)
decay_weights = np.array([1, 2, 0.1])
avg_denominator = decay_weights.sum()


class Simulation:
    id: int
    # total_profit: float
    total_history: int
    decays: npt.ArrayLike
    last_decay: int

    def __init__(self, id, last_decay):
        self.id = id
        self.last_decay = last_decay
        # self.total_profit = 1
        self.decays = np.array([1, 1, 1], dtype=np.float32)
        self.total_history = 0
        self.success_streak = 0

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, o: object) -> bool:
        return self.id == o.id

    def add_profit(self, profit, current_time):
        # self.total_profit *= profit
        elapsed = current_time - self.last_decay
        # for rate, val in self.decays.items():
        self.total_history += 1
        if profit >= 1.003:
            self.success_streak += 1
        elif profit < 1:
            self.success_streak = 0
        self.decays = get_decay(
            self.decays,
            decay_rates,
            profit,
            elapsed,
        )
        self.last_decay = current_time


class SimulationEncyclopedia:
    simulations: Dict[int, Simulation]

    def __init__(self, simulations: int, initial_decay_time):
        ids = range(simulations)
        self.simulations = dict.fromkeys(set(ids))
        for id in ids:
            self.simulations[id] = Simulation(id, initial_decay_time)

    def add_profits(self, sells, current_time, threshold_simulations, threshold):
        for sell in sells:
            index, timestamp, simulation_id, trigger_price, profit = sell
            sim = self.simulations.get(simulation_id)
            sim.add_profit(profit, current_time)
            if (
                sim.success_streak >= 3
                and sim.total_history > threshold["min_trade_history"]
                and sim.decays[2] >= threshold["min_profit"]
            ):
                threshold_simulations.add(sim)
            else:
                threshold_simulations.discard(sim)
        return threshold_simulations


cache = {}


# @timeme
def get_best_simulations(
    current_minute,
    simulations,
    margin: float = 0.99,
    min_profit=1.015,
    min_trade_history=3,
    min_success_ratio=0.8,
    pbar=None,
):
    profits = [
        (
            sim.id,
            # weighted_profit,
            # sim.total_profit,
            # sim.success_count,
            # sim.history_count,
            # sim.total_history_count,
            decay_avg,
            sim.decays[1],
            sim.decays,
        )
        # TODO: [df] loop through df
        for sim in simulations
        if (
            decay_avg := ((decay_weights / (avg_denominator)) * sim.decays).sum()
            # for idx, decayed_profit in enumerate(sim.decays)
        )
        > min_profit
        # decreases the effectiveness of caching sorted profits
        # if sim.history_count >= min_trade_history
        # and sim.total_history_count >= 20
        # and sim.success_count / sim.history_count >= min_success_ratio
        # and sim.total_profit >= min_profit
        # and (
        #     weighted_profit := (
        #         (sim.total_profit - 1) * (sim.success_count / sim.history_count)
        #     )
        #     + 1
        # )
        # >= min_profit
    ]
    if not len(profits):
        return []
    # else:
    #     if pbar is not None:
    #         pbar.write(f"[{now()}] ({current_minute}) profits ========= {len(profits)}")
    #     else:
    #         print(f"profits ========= {len(profits)}")
    best_profit = max(profits, key=itemgetter(1))[1]
    minimum = ((best_profit - min_profit) * margin) + min_profit
    best = [
        (key, decay_avg, decay, decays)
        for key, decay_avg, decay, decays in profits
        if decay_avg > minimum
    ]

    if len(best) > 0:
        best_sorted = sorted(best, key=itemgetter(2, 1), reverse=True)
        best_sorted_top_5 = pp.pformat(best_sorted[:5])
        if cache.get("best_sorted_top_5") != best_sorted_top_5:
            if pbar is not None:
                pbar.write(
                    f"[{now()}] ({current_minute}) qualifier {len(best)} {best_sorted_top_5}"
                )
            else:
                printd(f"({current_minute}) qualifier {len(best)} {best_sorted_top_5}")
            cache["best_sorted_top_5"] = best_sorted_top_5
        return [id for id, *_ in best_sorted]
    return []


class Portfolio:
    just_ordered: bool
    orders: List[Union[BuyOrder, SellOrder]]
    direction: Direction
    total_profit: float

    def __init__(self):
        self.orders = []
        self.direction = Direction.BUY
        self.total_profit = 1.0

    def find_order(self, best_simulations, current_order_df):
        if current_order_df is None:
            return None

        best_orders = current_order_df.loc[
            current_order_df["simulation_id"].isin(best_simulations)
        ]

        return best_orders.iloc[0] if not best_orders.empty else None

    def add_buy_order(self, best_simulations, current_order_df, pbar):
        order = self.find_order(best_simulations, current_order_df)

        if order is None:
            return None
        [timestamp, simulation_id, trigger_price] = order
        order = BuyOrder(timestamp, trigger_price, simulation_id)
        pbar.write(f"[{now()}] Buying your mom {order}")
        self.orders.append(order)
        self.direction = Direction.SELL
        self.just_ordered = True

    def add_sell_order(self, best_simulations, current_order_df, pbar):
        order = self.find_order(best_simulations, current_order_df)

        if order is None:
            return None

        [timestamp, simulation_id, trigger_price, profit] = order
        last_order: BuyOrder = self.orders[-1]
        profit = calculate_profit(last_order[1], trigger_price, buy_fee, sell_fee)
        order = SellOrder(timestamp, trigger_price, profit)
        self.total_profit *= profit
        pbar.write(f"[{now()}] Selling yo mama {order}, {self.total_profit=}")
        self.orders.append(order)
        self.direction = Direction.BUY
        self.just_ordered = True


class BuyLoader:
    location: pathlib.Path
    cached_buys: pdGroupby.GroupBy
    loaded_date: pd.Timestamp
    loaded_df: pd.DataFrame

    def __init__(self, location):
        self.location = location
        self.loaded_date = None

    def get_buys(self, date: pd.Timestamp):
        if (
            self.loaded_date is None
            or self.loaded_date.year != date.year
            or self.loaded_date.dayofyear != date.dayofyear
        ):
            self.cached_buys = load_parquet_by_date(self.location, date, "buy")
            self.cached_buys = self.cached_buys.iloc[
                self.cached_buys["timestamp"].searchsorted(date, side="left") :
            ]
            self.cached_buys = self.cached_buys.groupby("timestamp").__iter__()
            self.loaded_date, self.loaded_df = next(self.cached_buys, (None, None))

        if self.loaded_date is not None and self.loaded_date < date:
            self.loaded_date, self.loaded_df = next(self.cached_buys, (None, None))

        return self.loaded_df if self.loaded_date == date else None


@timeme
def main(pair, startDate, endDate, aroon_reference):
    order_dir = (
        pathlib.Path("/mnt/f/hydra") / order_folder
        if platform.system() == "Linux"
        else pathlib.Path("F:/hydra") / order_folder
    )
    output_dir = pathlib.Path.cwd().joinpath("output", pair)
    entry_ids = load_output_signal(
        output_dir / "signals", f"{aroon_reference} - aroon_entries.ids"
    )
    exit_ids = load_output_signal(
        output_dir / "signals", f"{aroon_reference} - aroon_exits.ids"
    )

    printd("Initializing Simulations")
    encyclopedia = SimulationEncyclopedia(len(entry_ids) * len(exit_ids), 0)

    printd("Preparing to simp-de-sim and pull")
    total_ticks = pd.to_datetime(endDate) - pd.to_datetime(startDate)
    total_ticks = total_ticks.total_seconds() / 60
    with tqdm(total=total_ticks, unit="tick", smoothing=0) as pbar:
        counter = 0
        threshold_simulations = set()
        portfolio = Portfolio()
        buy_loader = BuyLoader(order_dir)
        best_simulations = []
        for current_day in datetime_range(startDate, endDate, pd.Timedelta(1, "day")):
            try:
                sell_day = load_parquet_by_date(
                    order_dir,
                    current_day,
                    "sell",
                )
            except Exception as e:
                pbar.write(f"[{now()}] {e}")
                continue
            # sell_day.drop("trigger_price", axis=1, inplace=True)
            sells = sell_day.groupby("timestamp")
            sells = sells.__iter__()
            current_sell_date, current_sell_df = next(sells, (None, None))
            for current_minute in datetime_range(
                current_day,
                current_day + pd.Timedelta(1, "day"),
                pd.Timedelta(1, "minute"),
            ):
                pbar.update(1)
                counter += 1
                portfolio.just_ordered = False
                # counter == 1440:

                if current_sell_date == current_minute:
                    pbar.set_description(f"Sell Length {len(current_sell_df)}")
                    threshold_simulations = encyclopedia.add_profits(
                        current_sell_df.itertuples(name=None),
                        counter,
                        threshold_simulations,
                        threshold_options,
                    )

                    if portfolio.direction == Direction.SELL:
                        portfolio.add_sell_order(
                            best_simulations, current_sell_df, pbar=pbar
                        )

                    if portfolio.just_ordered:
                        best_simulations = get_best_simulations(
                            current_minute,
                            threshold_simulations,
                            **threshold_options,
                            pbar=pbar,
                        )

                    current_sell_date, current_sell_df = next(sells, (None, None))

                if not portfolio.just_ordered and portfolio.direction == Direction.BUY:
                    if counter % 5 == 0:
                        best_simulations = get_best_simulations(
                            current_minute,
                            threshold_simulations,
                            **threshold_options,
                            pbar=pbar,
                        )
                    if len(best_simulations):
                        buys = buy_loader.get_buys(current_minute)
                        portfolio.add_buy_order(best_simulations, buys, pbar=pbar)

        print(portfolio.orders)
        print(json.dumps(portfolio.orders))


main(pair, startDate, endDate, aroon_reference)
