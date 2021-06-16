import traceback
import sys
import os
import concurrent.futures
import cProfile
import itertools
import pathlib
from datetime import datetime, timedelta
from operator import itemgetter
from typing import Dict, List, NamedTuple, Set, Tuple, TypeVar

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import ray
from tqdm import tqdm

from hydra.models import Direction
from hydra.money import calculate_profit
from hydra.utils import now, printd, timeme

NUM_CHUNKS = psutil.cpu_count(logical=True)
current_time = datetime.now().strftime("%Y-%m-%dT%H%M")
pair = "XBTUSD"
startDate = pd.to_datetime("2017-01-01")
endDate = pd.to_datetime("2021-01-01")
reference_time = "2021-05-20T1919"
fee = 0.001
buy_fee = 1 + fee
sell_fee = 1 - fee
project_dir = pathlib.Path(__file__).absolute().parent.parent
saved_order_dir = project_dir / "output" / pair / "orders"

print(psutil.virtual_memory().percent)


def chunks(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]


class BuyOrder(NamedTuple):
    timestamp: datetime
    simulation_id: int
    trigger_price: float  # open if aroon is -1 else associated trigger price


class SellOrder(NamedTuple):
    timestamp: datetime
    simulation_id: int
    trigger_price: float
    profit: float
    hold_time: datetime


def load_output_signal(output_dir, reference_time, file) -> pd.DataFrame:
    path = output_dir / "signals" / f"{reference_time} - {file}.parquet"
    return pq.read_table(path).to_pandas()


class Simulation:
    id: int
    entry_id: int
    exit_id: int
    last_order: bool
    buy_trigger_price: float
    buy_time: datetime

    def __init__(self, id, entry_id, exit_id):
        self.id = id
        self.entry_id = entry_id
        self.exit_id = exit_id
        self.last_order = False
        self.buy_trigger_price = None
        self.buy_time = None

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, o: object) -> bool:
        return self.id == o.id


def get_profit(simulation, sell_trigger_price):
    return calculate_profit(
        simulation.buy_trigger_price,
        sell_trigger_price,
        buy_fee,
        sell_fee,
    )


def get_hold_time(simulation, sell_time):
    return sell_time - simulation.buy_time


def create_sell_order(simulation, timestamp, trigger, timestamp_in_minutes):
    order = SellOrder(
        timestamp,
        simulation.id,
        trigger,
        get_profit(simulation, trigger),
        get_hold_time(simulation, timestamp_in_minutes),
    )
    simulation.last_order = False
    simulation.buy_trigger_price = None
    simulation.buy_time = None
    return order


def create_buy_order(simulation, timestamp, trigger, timestamp_in_minutes):
    order = BuyOrder(timestamp, simulation.id, trigger)
    simulation.last_order = True
    simulation.buy_trigger_price = trigger
    simulation.buy_time = timestamp_in_minutes
    return order


# def update_buy(simulation, trigger, timestamp_in_minutes):
#     simulation.last_order = True
#     simulation.buy_trigger_price = trigger
#     simulation.buy_time = timestamp_in_minutes


# def update_sell(simulation):
#     simulation.last_order = False
#     simulation.buy_trigger_price = None
#     simulation.buy_time = None


@ray.remote(num_gpus=0)
class SimulationChunk:
    id: int
    simulations: Dict[int, Simulation]
    simulations_by_entry: Dict[int, Simulation]
    simulations_by_exit: Dict[int, Simulation]

    entry_timestamp_minutes: int
    exit_timestamp_minutes: int

    def __init__(self, id, simulations):
        self.id = id
        self.simulations = dict.fromkeys(set(simulations.keys()))
        self.simulations_by_entry = {}  # dict.fromkeys(set(entries))
        self.simulations_by_exit = {}  # dict.fromkeys(set(exits))

        self.startDate = startDate

        self.entry_timestamp_minute = 0
        self.exit_timestamp_minutes = 0

        self.next_entry = None
        self.next_exit = None
        for (id, val) in simulations.to_dict("index").items():
            # sim = {
            #     "id": id,
            #     "entry_id": val["entry_id"],
            #     "exit_id": val["exit_id"],
            #     "last_order": False,
            #     "buy_trigger_price": None,
            #     "buy_time": None,
            # }
            sim = Simulation(id, val["entry_id"], val["exit_id"])
            self.simulations[id] = sim
            self.simulations_by_entry.setdefault(val["entry_id"], set()).add(sim)
            self.simulations_by_exit.setdefault(val["exit_id"], set()).add(sim)

    # def update_buy(self, id, *args, **kwargs):
    #     return update_buy(self.simulations[id], *args, **kwargs)

    # def update_sell(self, id, *args, **kwargs):
    #     return update_sell(self.simulations[id], *args, **kwargs)

    def simulate_day(self, entries, exits, date, actor_id):
        buy_saver = OrderSaver("buy")
        sell_saver = OrderSaver("sell")

        self.entries = entries.__iter__()
        self.exits = exits.groupby("timestamp").__iter__()

        next_entry = next(self.entries)
        next_exit = next(self.exits)

        buys = []
        sells = []

        # print(date.day)
        # print(next_entry[0].day)
        # print(next_exit[0].day)
        while (next_entry[0] is not None and date.day == next_entry[0].day) or (
            next_exit[0] is not None and date.day == next_exit[0].day
        ):
            next_entry_timestamp, next_entry_df = next_entry
            if next_entry_timestamp is not None:
                timestamp_in_minutes_entry = (
                    next_entry_timestamp - self.startDate
                ).total_seconds() / 60

            next_exit_timestamp, next_exit_df = next_exit
            if next_exit_timestamp is not None:
                timestamp_in_minutes_exit = (
                    next_exit_timestamp - self.startDate
                ).total_seconds() / 60

            # If exits finish abort
            if next_exit_timestamp is None:
                break  # TODO: return some indicator that we are all out of dates/times?
            elif next_entry_timestamp == next_exit_timestamp:
                sells += self.create_orders(
                    next_exit_df, timestamp_in_minutes_exit, True
                )
                buys += self.create_orders(
                    next_entry_df, timestamp_in_minutes_entry, False
                )

                next_entry = next(self.entries, (None, None))
                next_exit = next(self.exits, (None, None))
            # if entries finish before exits, keep doing exits
            elif (
                next_entry_timestamp is None
                or next_exit_timestamp < next_entry_timestamp
            ):
                sells += self.create_orders(
                    next_exit_df, timestamp_in_minutes_exit, True
                )
                # encyclopedia.update_sells(sells)

                next_exit = next(self.exits, (None, None))
            elif next_entry_timestamp < next_exit_timestamp:
                buys = self.create_orders(
                    next_entry_df, timestamp_in_minutes_entry, False
                )
                # encyclopedia.update_buys(buys, timestamp_in_minutes_entry)

                next_entry = next(self.entries, (None, None))

        buy_saver.save_orders(buys, date, actor_id)

        sell_saver.save_orders(sells, date, actor_id)

        return None

    def create_orders(
        self,
        indicator_signal: pd.DataFrame,
        timestamp_in_minutes,
        is_exit,
    ) -> List[TypeVar("Order", BuyOrder, SellOrder)]:
        indicator_signal = indicator_signal.loc[indicator_signal["trigger"] != 0]
        if is_exit:
            return [
                create_sell_order(sim, timestamp, trigger, timestamp_in_minutes)
                for idx, timestamp, trigger, id in indicator_signal.itertuples()
                for sim in self.simulations_by_exit.get(id)
                if sim.last_order == Direction.BUY
            ]

        else:
            return [
                create_buy_order(sim, timestamp, trigger, timestamp_in_minutes)
                for idx, timestamp, trigger, id in indicator_signal.itertuples()
                for sim in self.simulations_by_entry.get(id)
                if sim.last_order == Direction.SELL
            ]


@timeme
def main():
    # pair, startDate, endDate, reference_time
    # printd("Loading Ids")
    output_dir = project_dir / "output" / pair
    # entry_ids = load_output_signal(output_dir, reference_time, "aroon_entries.ids")
    # exit_ids = load_output_signal(output_dir, reference_time, "aroon_exits.ids")

    printd("Loading Entries")
    entries = (
        load_output_signal(output_dir, reference_time, "aroon_entries")
        # .set_index("timestamp", "id")
    )
    entry_ids = entries["id"].unique()
    entries = entries.groupby(entries["timestamp"].dt.date)
    # for date, daily_entry in entries:
    #    todays_entry = daily_entry.groupby("timestamp")
    #    todays_entry = todays_entry.__iter__()
    #    next_entry_timestamp, next_entry_df = next(todays_entry)

    printd("Loading Exits")
    exits = (
        load_output_signal(output_dir, reference_time, "aroon_exits")
        # .set_index("timestamp", "id")
    )

    exit_ids = exits["id"].unique()
    exits = exits.groupby(exits["timestamp"].dt.date)

    printd("Preparing Simulations")
    simulations_df = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [entry_ids, exit_ids], names=["entry_id", "exit_id"]
        )
    )
    simulations_df = simulations_df.reset_index()
    simulations_df["chunk"] = simulations_df["exit_id"] % NUM_CHUNKS
    simulations_df = simulations_df.groupby("chunk")

    actor_chunks = {}
    for (sim_chunk, simulation) in tqdm(simulations_df, total=NUM_CHUNKS):
        actor_chunks[sim_chunk] = SimulationChunk.remote(sim_chunk, simulation)

    orders = {"buy": [], "sell": []}

    buy_saver = OrderSaver("buy")
    sell_saver = OrderSaver("sell")

    printd("Preparing to loop-de-loop and pull")

    total_days = endDate - startDate
    total_days = total_days.total_seconds() / 60 / 60 / 24
    order_futures_list = []
    with tqdm(total=total_days, unit="day", smoothing=0) as pbar:
        current_day = startDate
        entries = entries.__iter__()
        exits = exits.__iter__()
        entries_day, entries_day_df = next(entries)
        exits_day, exits_day_df = next(exits)
        while current_day < endDate:
            # pbar.set_description(str(current_day))
            # pbar.write(f"{current_day=} {entries_day=} {exits_day=}")
            while entries_day < current_day:
                entries_day, entries_day_df = next(entries)
            while exits_day < current_day:
                exits_day, exits_day_df = next(exits)

            chunked_exits_day_df = exits_day_df.groupby(exits_day_df["id"] % NUM_CHUNKS)

            entries_ref = ray.put(entries_day_df.groupby("timestamp"))  # of current day
            order_futures = [
                actor_chunks[actor_id]
                .simulate_day.remote(entries_ref, chunked_exit, current_day, actor_id)
                .future()
                for actor_id, chunked_exit in chunked_exits_day_df
            ]

            order_futures_list.append(order_futures)

            pbar.update(1)
            current_day += timedelta(days=1)
            # pbar.write(f"Moving onto day {current_day}")
    with tqdm(total=total_days, unit="day", smoothing=0) as pbar:
        for order_futures in order_futures_list:
            for future in concurrent.futures.as_completed(order_futures):
                if future.done():
                    pbar.update(1 / NUM_CHUNKS)


class OrderSaver:
    type: str
    columns: List[str]

    def __init__(self, type):
        self.type = type
        self.columns = BuyOrder._fields if type == "buy" else SellOrder._fields
        self.lastname = None
        self.writer = None

    def save_orders(self, orders, timestamp, core):
        # if (
        #     last_timestamp.day == next_timestamp.day
        #     or len(orders[self.type]) == 0
        #     # or counter % 360 == 0
        # ):
        #     return last_timestamp

        # pbar.set_description(f"[{now()}] Saving {timestamp.date()}")

        filename = f"day={timestamp.day}"
        order_df = pd.DataFrame.from_records(
            [order for order in orders],
            columns=self.columns,
        )
        order_df.sort_values("timestamp", axis=0, kind="mergesort", inplace=True)
        table = pa.Table.from_pandas(order_df)

        path = (
            saved_order_dir
            / f"order {current_time} fee={fee}"
            / f"year={timestamp.year}"
            / f"month={timestamp.month}"
            / f"core={core}"
        )

        # pbar.write(self.type, order_df)
        if not path.exists():
            path.mkdir(parents=True)
        if self.lastname is None or filename != self.lastname:
            if self.writer is not None:
                self.writer.close()
            self.lastname = filename
            self.writer = pq.ParquetWriter(
                path / f"{filename}.{self.type}.parquet",
                table.schema,
                compression="brotli",
                compression_level=2,
            )

        self.writer.write_table(table)
        # pbar.set_description(f"[{now()}] Saved {timestamp.date()}")
        # timestamp = next_timestamp
        # orders[self.type] = []
        # return timestamp


idx = pd.IndexSlice


def flatten(object):
    for item in object:
        if isinstance(item, (list, tuple, set)):
            yield from flatten(item)
        else:
            yield item


def filter_orders(buys, sells) -> Tuple[List[BuyOrder], List[SellOrder]]:
    buy_orders = {
        order for buy_future in buys if (order := buy_future.result()) is not None
    }
    sell_orders = {
        order for sell_future in sells if (order := sell_future.result()) is not None
    }
    both = {order[1] for order in buy_orders} & {order[1] for order in sell_orders}

    return (
        [buy for buy in buy_orders if buy[1] not in both],
        [sell for sell in sell_orders if sell[1] not in both],
    )


if __name__ == "__main__":
    try:
        ray.init(include_dashboard=False, local_mode=False)
        # cProfile.run("main()")
        main()
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
