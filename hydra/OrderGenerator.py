import os
from datetime import datetime
from hydra.utils import now, printd, timeme
import pathlib
from typing import Dict, List, NamedTuple, Set, Tuple, TypeVar, Union
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import numpy as np


def load_output_signal(output_dir, reference_time, file) -> pd.DataFrame:
    path = output_dir / "signals" / f"{reference_time} - {file}.parquet"
    return pq.read_table(path).to_pandas()


class BuyOrder(NamedTuple):
    timestamp: datetime
    simulation_id: int
    trigger_price: float  # open if aroon is -1 else associated trigger price


class SellOrder(NamedTuple):
    timestamp: datetime
    simulation_id: int
    trigger_price: float
    profit: float


class Simulation:
    id: int
    entry_id: int
    exit_id: int
    open_position: bool
    buy_trigger_price: float

    def __init__(self, id, entry_id, exit_id):
        self.id = id
        self.entry_id = entry_id
        self.exit_id = exit_id
        self.open_position = False
        self.buy_trigger_price = None
        pass

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, o: object) -> bool:
        return self.id == o.id

    def get_profit(self, sell_trigger_price):
        return sell_trigger_price / self.buy_trigger_price


class SimulationEncyclopedia:
    simulations: Dict[int, Simulation]
    by_entry: Dict[int, Set[Simulation]]
    by_exit: Dict[int, Set[Simulation]]

    def __init__(self, simulations: Dict[int, Dict]):
        self.by_entry = {}  # dict.fromkeys(set(entries))
        self.by_exit = {}  # dict.fromkeys(set(exits))
        self.simulations = dict.fromkeys(set(simulations.keys()))
        for id, val in simulations.items():
            sim = Simulation(id, val["entry_id"], val["exit_id"])
            self.simulations[id] = sim
            self.by_entry.setdefault(val["entry_id"], set()).add(sim)
            self.by_exit.setdefault(val["exit_id"], set()).add(sim)

    def update_buys(self, buys: List[BuyOrder]):
        for timestamp, simId, trigger in buys:
            sim = self.simulations.get(simId)
            sim.open_position = True
            sim.buy_trigger_price = trigger

    def update_sells(self, sells: List[SellOrder]):
        for timestamp, simId, trigger, profit in sells:
            sim = self.simulations.get(simId)
            sim.open_position = False
            sim.buy_trigger_price = None


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
    print(encyclopedia.simulations.get(0))
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
    last_entry_day = None
    last_exit_day = None

    total_ticks = pd.to_datetime(endDate) - pd.to_datetime(startDate)
    total_ticks = total_ticks.total_seconds() / 60
    with tqdm(total=total_ticks, unit="tick", smoothing=0) as pbar:
        entries = entries.__iter__()
        exits = exits.__iter__()
        next_entry_timestamp, next_entry_df = next(entries)
        next_exit_timestamp, next_exit_df = next(exits)
        # pbar.update(2)
        start_time = next_entry_timestamp
        counter = 0
        while True:
            # pbar.write(f"{next_entry_timestamp}, {next_exit_timestamp}")

            if counter % 10 == 0:
                next_time = next_entry_timestamp
                pbar.update((next_time - start_time).total_seconds() / 60)
                start_time = next_time
            counter += 1
            # snappy: 15:07
            # brotli2 / 10 counts: 15:37 (half the space)
            # brotli2 / 100 counts: 15:08
            # brotli2 / 360 counts: 15:06
            # brotli2 / 750 counts: 15:26
            # brotli2 / 1440 counts: 15:04

            if next_entry_timestamp < pd.to_datetime("2017-05-15 23:30"):
                continue
            # If exits finish abort
            if next_exit_timestamp is None or next_entry_timestamp > pd.to_datetime(
                "2017-05-16 00:30"
            ):
                break
            elif next_entry_timestamp == next_exit_timestamp:
                # pbar.write(f"[{now()}] Tiedstamp")
                next_entry_timestamp, next_entry_df = next(entries, (None, None))
                next_exit_timestamp, next_exit_df = next(exits, (None, None))

                last_entry_day = save_orders(
                    orders,
                    buy_saver,
                    "buys",
                    last_entry_day,
                    next_entry_timestamp,
                    pbar,
                )
                last_exit_day = save_orders(
                    orders,
                    sell_saver,
                    "sells",
                    last_exit_day,
                    next_exit_timestamp,
                    pbar,
                )

                # pbar.write(f"[{now()}] Creating Orders")
                sells = create_orders(encyclopedia, next_exit_df, True)
                buys = create_orders(encyclopedia, next_entry_df, False)

                # pbar.write(f"[{now()}] Filtering Orders")
                buys, sells = filter_orders(buys, sells)

                # pbar.write(f"[{now()}] Updating Simulations")
                encyclopedia.update_buys(buys)
                orders["buy"] += buys
                encyclopedia.update_sells(sells)
                orders["sell"] += sells
            elif next_entry_timestamp < next_exit_timestamp:
                # pbar.write(f"[{now()}] Lagging Entry")
                next_entry_timestamp, next_entry_df = next(entries, (None, None))

                last_entry_day = save_orders(
                    orders,
                    buy_saver,
                    "buys",
                    last_entry_day,
                    next_entry_timestamp,
                    pbar,
                )
                # pbar.write(f"[{now()}] Creating Orders")
                buys = create_orders(encyclopedia, next_entry_df, False)
                # pbar.write(f"[{now()}] Updating Simulations")
                encyclopedia.update_buys(buys)
                orders["buy"] += buys
            # if entries finish before exits, keep doing exits
            elif (
                next_entry_timestamp is None
                or next_exit_timestamp < next_entry_timestamp
            ):
                # pbar.write(f"[{now()}] Lagging Exit")
                next_exit_timestamp, next_exit_df = next(exits, (None, None))

                last_exit_day = save_orders(
                    orders,
                    sell_saver,
                    "sells",
                    last_exit_day,
                    next_exit_timestamp,
                    pbar,
                )
                # pbar.write(f"[{now()}] Creating Orders")
                sells = create_orders(encyclopedia, next_exit_df, True)
                # pbar.write(f"[{now()}] Updating Simulations")
                encyclopedia.update_sells(sells)
                orders["sell"] += sells


def save_orders(orders, saver, key, last_day, next_timestamp, pbar):
    if (
        last_day is None
        or last_day != next_timestamp.day
        # or counter % 360 == 0
    ):
        pbar.write(f"Saving {key}")
        saver.save_orders(
            f"day={last_day}",
            next_timestamp,
            orders[key],
        )
        last_day = next_timestamp.day
        orders[key] = []
    return last_day


class OrderSaver:
    type: str
    columns: List[str]

    def __init__(self, type):
        self.type = type
        self.columns = BuyOrder._fields if type == "buy" else SellOrder._fields
        self.lastname = None
        self.writer = None

    def save_orders(self, filename, timestamp, orders):
        if not len(orders):
            return
        order_df = pd.DataFrame.from_records(orders, columns=self.columns)
        table = pa.Table.from_pandas(order_df)

        path = pathlib.Path(
            f"F:/hydra/orders-fixd/year={timestamp.year}/month={timestamp.month}"
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


idx = pd.IndexSlice


def flatten(object):
    for item in object:
        if isinstance(item, (list, tuple, set)):
            yield from flatten(item)
        else:
            yield item


def create_orders(
    encyclopedia: SimulationEncyclopedia, indicator_signal, desired_position
) -> List[TypeVar("Order", BuyOrder, SellOrder)]:
    # printd("Creating order", desired_position)
    indicator_signal = indicator_signal.loc[indicator_signal["trigger"] != 0]
    # printd("Signal Indicated")

    # Exit
    if desired_position:
        level = "exit_id"
        orders = [
            SellOrder(timestamp, sim.id, trigger, sim.get_profit(trigger))
            for idx, timestamp, trigger, id in indicator_signal.itertuples()
            for sim in encyclopedia.by_exit.get(id)
            if sim.open_position == desired_position
        ]
    # Entry
    else:
        level = "entry_id"
        orders = [
            BuyOrder(timestamp, sim.id, trigger)
            for idx, timestamp, trigger, id in indicator_signal.itertuples()
            for sim in encyclopedia.by_entry.get(id)
            if sim.open_position == desired_position
        ]
    # printd("Simulations Found")
    # printd("Simulating Desire")
    return orders


def filter_orders(buys, sells) -> Tuple[List[BuyOrder], List[SellOrder]]:
    buysims = {simId for timestamp, simId, trigger in buys}
    sellsims = {simId for timestamp, simId, trigger, profit in sells}
    both = buysims & sellsims

    return [buy for buy in buys if buy[1] not in both], [
        sell for sell in sells if sell[1] not in both
    ]


pair = "XBTUSD"
startDate = "2017-05-15"
endDate = "2021-06-16"
reference_time = "2021-05-20T1919"
main(pair, startDate, endDate, reference_time)
# groupless(pair, startDate, endDate, reference_time)