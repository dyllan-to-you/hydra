import concurrent.futures
import cProfile
import pathlib
from datetime import datetime
import itertools
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

NUM_CHUNKS = psutil.cpu_count(logical=False)
current_time = datetime.now().strftime("%Y-%m-%dT%H%M")
pair = "XBTUSD"
startDate = "2018-04-12"
endDate = "2021-01-01"
reference_time = "2021-05-20T1919"
fee = 0.001
buy_fee = 1 + fee
sell_fee = 1 - fee
project_dir = pathlib.Path(__file__).absolute().parent.parent
saved_order_dir = project_dir / "output" / pair / "orders"


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
    open_position: bool
    buy_trigger_price: float
    buy_time: datetime

    def __init__(self, id, entry_id, exit_id):
        self.id = id
        self.entry_id = entry_id
        self.exit_id = exit_id
        self.open_position = False
        self.buy_trigger_price = None
        self.buy_time = None

    # def __hash__(self) -> int:
    #     return self.id

    # def __eq__(self, o: object) -> bool:
    #     return self.id == o.id


def get_profit(simulation, sell_trigger_price):
    return calculate_profit(
        simulation["buy_trigger_price"],
        sell_trigger_price,
        buy_fee,
        sell_fee,
    )


def get_hold_time(simulation, sell_time):
    return sell_time - simulation["buy_time"]


def create_sell_order(simulation, timestamp, trigger, timestamp_in_minutes):
    if simulation["open_position"] == Direction.BUY:
        order = SellOrder(
            timestamp,
            simulation["id"],
            trigger,
            get_profit(simulation, trigger),
            get_hold_time(simulation, timestamp_in_minutes),
        )
        simulation["open_position"] = False
        simulation["buy_trigger_price"] = None
        simulation["buy_time"] = None
        return order


def create_buy_order(simulation, timestamp, trigger, timestamp_in_minutes):
    if simulation["open_position"] == Direction.SELL:
        order = BuyOrder(timestamp, simulation["id"], trigger)
        simulation["open_position"] = True
        simulation["buy_trigger_price"] = trigger
        simulation["buy_time"] = timestamp_in_minutes
        return order


# def update_buy(simulation, trigger, timestamp_in_minutes):
#     simulation["open_position"] = True
#     simulation["buy_trigger_price"] = trigger
#     simulation["buy_time"] = timestamp_in_minutes


# def update_sell(simulation):
#     simulation["open_position"] = False
#     simulation["buy_trigger_price"] = None
#     simulation["buy_time"] = None


@ray.remote(num_gpus=0)
class SimulationChunk:
    simulations: Dict[int, Simulation]

    def __init__(self):
        self.simulations = dict()

    def add_simulation(self, simulation):
        self.simulations[simulation["id"]] = simulation

    def set_simulations(self, simulations):
        self.simulations = simulations

    # def update_buy(self, id, *args, **kwargs):
    #     return update_buy(self.simulations[id], *args, **kwargs)

    # def update_sell(self, id, *args, **kwargs):
    #     return update_sell(self.simulations[id], *args, **kwargs)

    def create_buy_order(self, id, *args, **kwargs):
        return create_buy_order(self.simulations[id], *args, **kwargs)

    def create_sell_order(self, id, *args, **kwargs):
        return create_sell_order(self.simulations[id], *args, **kwargs)

    def create_buy_orders(self, params, timestamp_in_min):
        return [
            create_buy_order(self.simulations[id], timestamp, trigger, timestamp_in_min)
            for (id, timestamp, trigger) in params
        ]

    def create_sell_orders(self, params, timestamp_in_min):
        return [
            create_sell_order(
                self.simulations[id], timestamp, trigger, timestamp_in_min
            )
            for (id, timestamp, trigger) in params
        ]


class SimulationEncyclopedia:
    chunks: List[SimulationChunk]
    simulation_chunks: Dict[int, SimulationChunk]
    by_entry: Dict[int, Set[int]]
    by_exit: Dict[int, Set[int]]

    def __init__(self, simulations: Dict[int, Dict]):
        self.by_entry = {}  # dict.fromkeys(set(entries))
        self.by_exit = {}  # dict.fromkeys(set(exits))
        self.simulation_chunks = dict.fromkeys(set(simulations.keys()))

        self.chunks = [SimulationChunk.remote() for chunk in range(NUM_CHUNKS)]

        prechunks = [dict() for chunk in range(NUM_CHUNKS)]
        for idx, (id, val) in enumerate(tqdm(simulations.items())):
            chunk_idx = idx % NUM_CHUNKS
            prechunks[chunk_idx][id] = {
                "id": id,
                "entry_id": val["entry_id"],
                "exit_id": val["exit_id"],
                "open_position": False,
                "buy_trigger_price": None,
                "buy_time": None,
            }
            self.simulation_chunks[id] = chunk_idx
            self.by_entry.setdefault(val["entry_id"], set()).add(id)
            self.by_exit.setdefault(val["exit_id"], set()).add(id)

        for chunk_idx, chunk in enumerate(tqdm(self.chunks)):
            chunk.set_simulations.remote(prechunks[chunk_idx])

    def get_simulation_chunk(self, id):
        return self.chunks[self.simulation_chunks[id]]

    # def update_buys(self, buys: List[BuyOrder], timestamp_in_minutes):
    #     # TODO: May need to collect "completed"
    #     # and check for completions across all simulations before returning
    #     is_future = None
    #     for buy_future in buys:
    #         if is_future is None:
    #             is_future = isinstance(buy_future, concurrent.futures.Future)
    #         timestamp, simId, trigger, *_ = (
    #             buy_future.result() if is_future else buy_future
    #         )

    #         self.get_simulation_chunk(simId).update_buy.remote(
    #             simId, trigger, timestamp_in_minutes
    #         )

    # def update_sells(self, sells: List[SellOrder]):
    #     is_future = None
    #     for sell_future in sells:
    #         if is_future is None:
    #             is_future = isinstance(sell_future, concurrent.futures.Future)
    #         timestamp, simId, *_ = sell_future.result() if is_future else sell_future
    #         self.get_simulation_chunk(simId).update_sell.remote(simId)

    def create_orders(
        self,
        indicator_signal: pd.DataFrame,
        timestamp_in_minutes,
        is_exit,
    ) -> List[TypeVar("Order", BuyOrder, SellOrder)]:
        # printd("Creating order", is_exit)
        indicator_signal = indicator_signal.loc[indicator_signal["trigger"] != 0]
        # printd("Signal Indicated")

        orders = dict()
        # Exit
        if is_exit:
            for idx, timestamp, trigger, id in indicator_signal.itertuples():
                for simId in self.by_exit.get(id):
                    chunk = self.simulation_chunks[simId]
                    orders[chunk] = orders.setdefault(chunk, list())
                    orders[chunk].append((simId, timestamp, trigger))

            orders = [
                self.chunks[chunk]
                .create_sell_orders.remote(order_params, timestamp_in_minutes)
                .future()
                for chunk, order_params in orders.items()
            ]

        # Entry
        else:
            for idx, timestamp, trigger, id in indicator_signal.itertuples():
                for simId in self.by_entry.get(id):
                    chunk = self.simulation_chunks[simId]
                    orders[chunk] = orders.setdefault(chunk, list())
                    orders[chunk].append((simId, timestamp, trigger))

            orders = [
                self.chunks[chunk]
                .create_buy_orders.remote(order_params, timestamp_in_minutes)
                .future()
                for chunk, order_params in orders.items()
            ]

            # sort simIds by chunk
            # then pass blocks to actor
        # printd("Simulations Found")
        # TODO: Verify that `done` does not contain any None/empty orders
        return concurrent.futures.as_completed(orders)


@timeme
def main():
    # pair, startDate, endDate, reference_time
    printd("Loading Ids")
    output_dir = pathlib.Path.cwd().joinpath("output", pair)
    entry_ids = load_output_signal(output_dir, reference_time, "aroon_entries.ids")
    exit_ids = load_output_signal(output_dir, reference_time, "aroon_exits.ids")

    printd("Preparing Simulations")
    simulations_df = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [entry_ids["id"], exit_ids["id"]], names=["entry_id", "exit_id"]
        )
    )
    simulations_df = simulations_df.reset_index()
    simulations_dick = simulations_df.to_dict("index")

    printd("Initializing Simulations")
    encyclopedia = SimulationEncyclopedia(simulations_dick)
    del simulations_df
    del simulations_dick
    printd("Loading Entries")
    entries = (
        load_output_signal(output_dir, reference_time, "aroon_entries")
        # .set_index("timestamp", "id")
    )
    entries = entries.groupby("timestamp")

    printd("Loading Exits")
    exits = (
        load_output_signal(output_dir, reference_time, "aroon_exits")
        # .set_index("timestamp", "id")
    )
    exits = exits.groupby("timestamp")

    orders = {"buy": [], "sell": []}

    buy_saver = OrderSaver("buy")
    sell_saver = OrderSaver("sell")

    printd("Preparing to loop-de-loop and pull")

    # entries = ray.put(entries)
    # exits = ray.put(exits)
    # SAVE entries and exits to plasma store

    # for chunk in encyclopedia.chunks:
    #     chunk.main.remote(entries, exits)

    total_ticks = pd.to_datetime(endDate) - pd.to_datetime(startDate)
    total_ticks = total_ticks.total_seconds() / 60
    with tqdm(total=total_ticks, unit="tick", smoothing=0) as pbar:
        entries = entries.__iter__()
        exits = exits.__iter__()
        next_entry_timestamp, next_entry_df = next(entries)
        next_exit_timestamp, next_exit_df = next(exits)
        last_entry_timestamp = next_entry_timestamp
        last_exit_timestamp = next_exit_timestamp
        # pbar.update(2)
        start_time_entry = next_entry_timestamp
        start_time_exit = next_exit_timestamp
        timestamp_in_minutes_entry = 0
        timestamp_in_minutes_exit = 0
        while True:
            # pbar.write(f"{next_entry_timestamp}, {next_exit_timestamp}")
            next_time_entry = next_entry_timestamp
            time_delta = (next_time_entry - start_time_entry).total_seconds() / 60
            pbar.update(time_delta)
            timestamp_in_minutes_entry += time_delta
            start_time_entry = next_time_entry

            next_time_exit = next_exit_timestamp
            time_delta = (next_time_exit - start_time_exit).total_seconds() / 60
            timestamp_in_minutes_exit += time_delta
            start_time_exit = next_time_exit
            # snappy: 15:07
            # brotli2 / 10 counts: 15:37 (half the space)
            # brotli2 / 100 counts: 15:08
            # brotli2 / 360 counts: 15:06
            # brotli2 / 750 counts: 15:26
            # brotli2 / 1440 counts: 15:04

            # If exits finish abort
            if next_exit_timestamp is None:
                break
            elif next_entry_timestamp == next_exit_timestamp:
                # pbar.write(f"[{now()}] Tiedstamp")
                last_entry_timestamp = buy_saver.save_orders(
                    orders,
                    last_entry_timestamp,
                    next_entry_timestamp,
                    pbar,
                )
                last_exit_timestamp = sell_saver.save_orders(
                    orders,
                    last_exit_timestamp,
                    next_exit_timestamp,
                    pbar,
                )

                # pbar.write(f"[{now()}] Creating Orders")
                sells = encyclopedia.create_orders(
                    next_exit_df, timestamp_in_minutes_exit, True
                )
                buys = encyclopedia.create_orders(
                    next_entry_df, timestamp_in_minutes_entry, False
                )

                # pbar.write(f"[{now()}] Filtering Orders")
                # buys, sells = filter_orders(buys, sells)

                # pbar.write(f"[{now()}] Updating Simulations")
                # encyclopedia.update_buys(buys, timestamp_in_minutes_entry)
                orders["buy"].append(buys)
                # encyclopedia.update_sells(sells)
                orders["sell"].append(sells)

                next_entry_timestamp, next_entry_df = next(entries, (None, None))
                next_exit_timestamp, next_exit_df = next(exits, (None, None))
            elif next_entry_timestamp < next_exit_timestamp:
                # pbar.write(f"[{now()}] Lagging Entry")
                last_entry_timestamp = buy_saver.save_orders(
                    orders,
                    last_entry_timestamp,
                    next_entry_timestamp,
                    pbar,
                )
                # pbar.write(f"[{now()}] Creating Orders")
                buys = encyclopedia.create_orders(
                    next_entry_df, timestamp_in_minutes_entry, False
                )
                # pbar.write(f"[{now()}] Updating Simulations")
                # encyclopedia.update_buys(buys, timestamp_in_minutes_entry)
                orders["buy"].append(buys)

                next_entry_timestamp, next_entry_df = next(entries, (None, None))
            # if entries finish before exits, keep doing exits
            elif (
                next_entry_timestamp is None
                or next_exit_timestamp < next_entry_timestamp
            ):
                # pbar.write(f"[{now()}] Lagging Exit")
                last_exit_timestamp = sell_saver.save_orders(
                    orders,
                    last_exit_timestamp,
                    next_exit_timestamp,
                    pbar,
                )
                # pbar.write(f"[{now()}] Creating Orders")
                sells = encyclopedia.create_orders(
                    next_exit_df, timestamp_in_minutes_exit, True
                )
                # pbar.write(f"[{now()}] Updating Simulations")
                # encyclopedia.update_sells(sells)
                orders["sell"].append(sells)

                next_exit_timestamp, next_exit_df = next(exits, (None, None))


class OrderSaver:
    type: str
    columns: List[str]

    def __init__(self, type):
        self.type = type
        self.columns = BuyOrder._fields if type == "buy" else SellOrder._fields
        self.lastname = None
        self.writer = None

    def save_orders(self, orders, last_timestamp, next_timestamp, pbar):
        if (
            last_timestamp.day == next_timestamp.day
            or len(orders[self.type]) == 0
            # or counter % 360 == 0
        ):
            return last_timestamp

        pbar.set_description(f"[{now()}] Saving {last_timestamp.date()}")

        filename = f"day={last_timestamp.day}"
        order_df = pd.DataFrame.from_records(
            [
                order.result()
                for order in itertools.chain.from_iterable(orders[self.type])
            ],
            columns=self.columns,
        )
        table = pa.Table.from_pandas(order_df)

        path = (
            saved_order_dir
            / f"order {current_time} fee={fee}"
            / f"year={last_timestamp.year}"
            / f"month={last_timestamp.month}"
        )

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
        pbar.set_description(f"[{now()}] Saved {last_timestamp.date()}")
        last_timestamp = next_timestamp
        orders[self.type] = []
        return last_timestamp


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
        ray.init(include_dashboard=False)
        cProfile.run("main()")
        # main()
    finally:
        ray.shutdown()
