import copy
import platform
import pathlib
from datetime import datetime
from typing import Dict, Set
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from hydra.utils import now, printd, timeme
from numba import njit


pair = "XBTUSD"
startDate = pd.to_datetime("2017-05-15")
endDate = pd.to_datetime("2021-06-16")
endDate = pd.to_datetime("2017-05-22")
aroon_reference = "2021-05-20T1919"
order_folder = "orders 2021-05-22T0041 fee=0.001"


def load_output_signal(output_dir, file) -> pd.DataFrame:
    path = output_dir / f"{file}.parquet"
    return pq.read_table(path).to_pandas()


def load_parquet_by_date(dir, year, month, day, name):
    parquet_dir = f"year={year}/month={month}"
    sell_filename = f"day={day}.{name}"
    return load_output_signal(dir / parquet_dir, sell_filename)


def datetime_range(start, end, delta, inclusive=False):
    current = start
    while current < end:
        yield current
        current += delta
    if inclusive:
        yield current


@njit(cache=True)
def get_decay(original, rate, profit, minutes_elapsed=1):
    return ((original * profit - 1) * (rate ** minutes_elapsed)) + 1


decay_rates = np.array(
    [
        0.05 ** (1 / 60),
        0.05 ** (1 / 180),
        0.05 ** (1 / 360),
        0.05 ** (1 / 720),
        0.05 ** (1 / 1440),
        0.05 ** (1 / 10080),
        0.05 ** (1 / 40320),
        0.05 ** (1 / 525600),
    ]
)


class Simulation:
    id: int
    total_profit: float
    decays: Dict[str, int]
    last_decay: int

    def __init__(self, id, last_decay):
        self.id = id
        self.last_decay = last_decay
        self.total_profit = 1
        self.decays = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, o: object) -> bool:
        return self.id == o.id

    def add_profit(self, profit, current_time):
        # self.total_profit *= profit
        elapsed = current_time - self.last_decay
        # for rate, val in self.decays.items():
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

    def add_profits(self, sells, current_time):
        for sell in sells:
            index, timestamp, simulation_id, profit = sell
            sim = self.simulations.get(simulation_id)
            sim.add_profit(profit, current_time)


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
        for current_day in datetime_range(startDate, endDate, pd.Timedelta(1, "day")):
            try:
                sell_day = load_parquet_by_date(
                    order_dir,
                    current_day.year,
                    current_day.month,
                    current_day.day,
                    "sell",
                )
            except Exception as e:
                pbar.write(f"{e}")
                continue
            sell_day.drop("trigger_price", axis=1, inplace=True)
            sells = sell_day.groupby("timestamp")
            sells = sells.__iter__()
            current_sell_date, current_sell_df = next(sells)
            for current_minute in datetime_range(
                current_day,
                current_day + pd.Timedelta(1, "day"),
                pd.Timedelta(1, "minute"),
            ):
                pbar.update(1)
                counter += 1
                if counter % 1440 == 0:
                    return

                if current_minute < current_sell_date:
                    continue
                pbar.set_description(f"Sell Length {len(current_sell_df)}")
                encyclopedia.add_profits(current_sell_df.itertuples(name=None), counter)

                current_sell_date, current_sell_df = next(sells)


main(pair, startDate, endDate, aroon_reference)
