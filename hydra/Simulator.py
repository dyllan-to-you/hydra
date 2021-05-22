import platform
import pathlib
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


decays = {
    60: 0.05 ** (1 / 60),
    180: 0.05 ** (1 / 180),
    360: 0.05 ** (1 / 360),
    720: 0.05 ** (1 / 720),
    1440: 0.05 ** (1 / 1440),
    10080: 0.05 ** (1 / 10080),
    40320: 0.05 ** (1 / 40320),
    525600: 0.05 ** (1 / 525600),
}


@njit
def get_decay(original, profit, rate, minutes_elapsed=1):
    return ((original * profit - 1) * (rate ** minutes_elapsed)) + 1


def add_sim_profit(sim, current_time):
    sim["total_profit"] *= sim["profit"]
    decay_keys = {
        "decay_60": 60,
        "decay_180": 180,
        "decay_360": 360,
        "decay_720": 720,
        "decay_1440": 1440,
        "decay_10080": 10080,
        "decay_40320": 40320,
        "decay_525600": 525600,
    }
    elapsed = (current_time - sim["last_decay"]) / np.timedelta64(1, "m")
    for decay, interval in decay_keys.items():
        sim[decay] = get_decay(
            sim[decay].to_numpy(),
            sim["profit"].to_numpy(),
            decays[interval],
            elapsed.to_numpy(),
        )
    sim["last_decay"] = current_time
    return sim


def add_sim_profit_orig(sim, profit, minutes_elapsed=1):
    sim["total_profit"] *= profit
    decay_keys = {
        "decay_60": 60,
        "decay_180": 180,
        "decay_360": 360,
        "decay_720": 720,
        "decay_1440": 1440,
        "decay_10080": 10080,
        "decay_40320": 40320,
        "decay_525600": 525600,
    }
    for decay, interval in decay_keys.items():
        sim[decay] = get_decay(sim[decay], profit, decays[interval])
    return sim


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

    # buy_filename = f"day={startDate.day}.buy"
    # buys = load_output_signal(order_dir, buy_filename)

    simulations_df = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [entry_ids["id"], exit_ids["id"]], names=["entry_id", "exit_id"]
        )
    )
    simulations_df = simulations_df.reset_index().drop(["entry_id", "exit_id"], axis=1)
    simulations_df.index.name = "id"
    simulations_df["total_profit"] = 1
    simulations_df["decay_60"] = 1
    simulations_df["decay_180"] = 1
    simulations_df["decay_360"] = 1
    simulations_df["decay_720"] = 1
    simulations_df["decay_1440"] = 1
    simulations_df["decay_10080"] = 1
    simulations_df["decay_40320"] = 1
    simulations_df["decay_525600"] = 1
    simulations_df["last_decay"] = startDate
    print(simulations_df)

    printd("Preparing to loop-de-loop and pull")
    total_ticks = pd.to_datetime(endDate) - pd.to_datetime(startDate)
    total_ticks = total_ticks.total_seconds() / 60
    with tqdm(total=total_ticks, unit="tick", smoothing=0) as pbar:
        # counter = 0
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
                # if counter % 10 == 0:
                # counter += 1

                if current_minute < current_sell_date:
                    continue
                pbar.set_description(f"Sell Length {len(current_sell_df)}")
                current_sell_df = current_sell_df.set_index("simulation_id")
                current_sell_df = current_sell_df.join(simulations_df, how="left")

                current_sell_df = add_sim_profit(current_sell_df, current_minute)
                simulations_df.update(current_sell_df)
                print(simulations_df)
                return
                current_sell_date, current_sell_df = next(sells)


main(pair, startDate, endDate, aroon_reference)
