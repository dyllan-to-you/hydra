from hydra.models import Direction
import pathlib
from datetime import datetime
import concurrent.futures
from typing import Dict, List, NamedTuple, Set, Tuple, TypeVar

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import psutil
import ray

from hydra.utils import now, printd, timeme
from hydra.money import calculate_profit


NUM_CHUNKS = psutil.cpu_count()
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

    def get_profit(self, sell_trigger_price):
        return calculate_profit(
            self.buy_trigger_price,
            sell_trigger_price,
            buy_fee,
            sell_fee,
        )

    def get_hold_time(self, sell_time):
        return sell_time - self.buy_time

    def create_sell_order(self, timestamp, trigger, timestamp_in_minutes):
        return (
            SellOrder(
                timestamp,
                self.id,
                trigger,
                self.get_profit(trigger),
                self.get_hold_time(timestamp_in_minutes),
            )
            if self.open_position == Direction.BUY
            else None
        )

    def create_buy_order(self, timestamp, trigger):
        return (
            BuyOrder(timestamp, self.id, trigger)
            if self.open_position == Direction.SELL
            else None
        )

    def update_buy(self, trigger, timestamp_in_minutes):
        self.open_position = True
        self.buy_trigger_price = trigger
        self.buy_time = timestamp_in_minutes

    def update_sell(self):
        self.open_position = False
        self.buy_trigger_price = None
        self.buy_time = None


@ray.remote
class SimulationActor:
    # simulations: Dict[int, Simulation]

    # def __init__(self):
    #     self.simulations = dict()

    # def add_simulation(self, simulation):
    #     self.simulations[simulation.id] = simulation

    def update_buy(self, id, *args, **kwargs):
        return ray.get(id).update_buy(*args, **kwargs)

    def update_sell(self, id, *args, **kwargs):
        return ray.get(id).update_sell(*args, **kwargs)

    def create_buy_order(self, id, *args, **kwargs):
        return ray.get(id).create_buy_order(*args, **kwargs)

    def create_sell_order(self, id, *args, **kwargs):
        return ray.get(id).create_sell_order(*args, **kwargs)


class SimulationEncyclopedia:
    actors: List[SimulationActor]
    actor_pool: ray.util.ActorPool
    simulations: Dict[int, Simulation]
    by_entry: Dict[int, Set[int]]
    by_exit: Dict[int, Set[int]]

    def __init__(self, simulations: Dict[int, Dict]):
        self.by_entry = {}  # dict.fromkeys(set(entries))
        self.by_exit = {}  # dict.frogmkeys(set(exits))
        self.simulations = dict.fromkeys(set(simulations.keys()))

        self.actors = [SimulationActor.remote() for chunk in range(NUM_CHUNKS)]
        self.actor_pool = ray.util.ActorPool(self.actors)

        for idx, (id, val) in enumerate(tqdm(simulations.items())):
            # chunk = self.actors[idx % NUM_CHUNKS]
            sim = ray.put(Simulation(id, val["entry_id"], val["exit_id"]))
            self.simulations[id] = sim
            self.by_entry.setdefault(val["entry_id"], set()).add(id)
            self.by_exit.setdefault(val["exit_id"], set()).add(id)

    def update_buys(self, buys: List[BuyOrder], timestamp_in_minutes):
        def fn(actor, value):
            if value is None:
                return None
            timestamp, simId, trigger, *_ = value
            return actor.update_buy.remote(
                self.simulations[simId], trigger, timestamp_in_minutes
            )

        printd("Updated", len(list(self.actor_pool.map_unordered(fn, buys))), "buys")

    def update_sells(self, sells: List[SellOrder]):
        def fn(actor, value):
            if value is None:
                return None
            timestamp, simId, trigger, *_ = value
            return actor.update_sell.remote(self.simulations[simId])

        printd("Updated", len(list(self.actor_pool.map_unordered(fn, sells))), "sells")

    def create_orders(
        self,
        indicator_signal: pd.DataFrame,
        timestamp_in_minutes,
        is_exit,
    ) -> List[TypeVar("Order", BuyOrder, SellOrder)]:
        # printd("Creating order", is_exit)
        indicator_signal = indicator_signal.loc[indicator_signal["trigger"] != 0]
        # printd("Signal Indicated")

        # Exit
        if is_exit:

            def create_sells(actor, value):
                simId, timestamp, trigger = value
                return actor.create_sell_order.remote(
                    self.simulations[simId], timestamp, trigger, timestamp_in_minutes
                )

            orders = self.actor_pool.map_unordered(
                create_sells,
                [
                    (simId, timestamp, trigger)
                    for idx, timestamp, trigger, id in indicator_signal.itertuples()
                    for simId in self.by_exit.get(id)
                ],
            )
        # Entry
        else:

            def create_buys(actor, value):
                simId, timestamp, trigger = value
                return actor.create_buy_order.remote(
                    self.simulations[simId], timestamp, trigger
                )

            orders = self.actor_pool.map_unordered(
                create_buys,
                [
                    (simId, timestamp, trigger)
                    for idx, timestamp, trigger, id in indicator_signal.itertuples()
                    for simId in self.by_entry.get(id)
                ],
            )
        # printd("Simulations Found")
        # printd("Simulating Desire")
        return filter(lambda x: x is not None, orders)


@timeme
def main(pair, startDate, endDate, reference_time):
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
                buys, sells = filter_orders(buys, sells)

                # pbar.write(f"[{now()}] Updating Simulations")
                encyclopedia.update_buys(buys, timestamp_in_minutes_entry)
                orders["buy"] += buys
                encyclopedia.update_sells(sells)
                orders["sell"] += sells

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
                encyclopedia.update_buys(buys, timestamp_in_minutes_entry)
                orders["buy"] += buys

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
                encyclopedia.update_sells(sells)
                orders["sell"] += sells

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
        order_df = pd.DataFrame.from_records(orders[self.type], columns=self.columns)
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
    buysims = {simId for timestamp, simId, trigger in buys}
    sellsims = {simId for timestamp, simId, *_ in sells}
    both = buysims & sellsims

    return [buy for buy in buys if buy[1] not in both], [
        sell for sell in sells if sell[1] not in both
    ]


if __name__ == "__main__":
    try:
        ray.init()
        main(pair, startDate, endDate, reference_time)
    finally:
        ray.shutdown()
