from tqdm.std import tqdm
import math
import sqlite3
import pathlib
import numpy as np
from numba import njit
import pandas as pd
import vectorbt as vbt
from vectorbt.signals.factory import SignalFactory
from dataloader.kraken import load_prices
import pyarrow as pa
import pyarrow.parquet as pq
from hydra.utils import timeme, now, printd
import gc
from time import time
from datetime import datetime
import os, psutil
import scipy as sp
import sparse

name = "Aroon"
current_time = datetime.now().strftime("%Y-%m-%dT%H%M")


def save_db(conn, df, table, *args, pbar=None, **kwargs):
    pbar.write(f"[{now()}] Converting dataframe to table: {table}")
    df.to_sql(f"{table}_df", conn, if_exists="replace", index=False, chunksize=100000)
    try:
        db_stmnt = f"INSERT INTO {table} SELECT * FROM {table}_df;"
        # pbar.write(f"[{now()}] {db_stmnt=}")
        conn.execute(db_stmnt)
    except sqlite3.OperationalError as e:
        if "no such table:" not in str(e):
            print(e)
            raise e
        db_stmnt = f"""
            CREATE TABLE {table} AS SELECT * FROM {table}_df;
        """
        # pbar.write(f"[{now()}] {db_stmnt=}")
        conn.executescript(db_stmnt)

    pbar.write(f"[{now()}] Committing db changes")
    conn.commit()


def load_multiple_prices(*args):
    prices = {}
    prices[1] = load_prices(*args, interval=1)
    prices[5] = load_prices(*args, interval=5)
    prices[15] = load_prices(*args, interval=15)
    prices[60] = load_prices(*args, interval=60)
    prices[720] = load_prices(*args, interval=720)
    prices[1440] = load_prices(*args, interval=1440)
    return prices


@njit
def aroon_entry(from_i, to_i, col, a, entry_threshold, temp_idx_arr):
    # if first time being called
    if from_i == 0:
        # build an array of the indices where the arron oscillation > entry_threshold
        w = np.where(a[:, col] > entry_threshold)[0]
        # save the array to temp_idx_arr
        for i, num in enumerate(w):
            temp_idx_arr[i] = num

    # for each potential index where aroon oscillation > entry threshold
    for i in range(len(temp_idx_arr)):
        # if the index is after the last exit (and before whatever to_i is (the next entry?? End of line??))
        if temp_idx_arr[i] > from_i and temp_idx_arr[i] < to_i:
            # return the index of the next entry
            return temp_idx_arr[i : i + 1]
    # return empty array
    return temp_idx_arr[:0]


@njit
def aroon_exit(from_i, to_i, col, a, exit_threshold, temp_idx_arr):
    # If it is the first time calling this function
    if temp_idx_arr[-1] != 42:
        # set our 'flag' saying we've called this before
        temp_idx_arr[-1] = 42
        # get all indices where the aroonoscillation < exit_threshold
        w = np.where(a[:, col] < exit_threshold)[0]
        # save all these indices to temp_idx_arr
        for i, num in enumerate(w):
            temp_idx_arr[i] = num
    # for each index
    for i in range(len(temp_idx_arr)):
        # if the index is after the last entry and before `to_i` (end of the simulation??)
        if temp_idx_arr[i] > from_i and temp_idx_arr[i] < to_i:
            # return the index for the next exit
            return temp_idx_arr[i : i + 1]
    return temp_idx_arr[:0]


AroonStrategy = SignalFactory(
    input_names=["aroon"], param_names=["entry_threshold", "exit_threshold"]
).from_choice_func(
    entry_choice_func=aroon_entry,
    entry_settings=dict(
        pass_inputs=["aroon"],
        pass_params=["entry_threshold"],
        pass_kwargs=["temp_idx_arr"],  # built-in kwarg
    ),
    exit_choice_func=aroon_exit,
    exit_settings=dict(
        pass_inputs=["aroon"],
        pass_params=["exit_threshold"],
        pass_kwargs=["temp_idx_arr"],  # built-in kwarg
    ),
    # forward_flex_2d=True,
)

AROONOSC = vbt.IndicatorFactory.from_talib("AROONOSC")
AROON = vbt.IndicatorFactory.from_talib("AROON")
# printd(help(AROONOSC.run))


def generate_signals(
    prices, timeperiods, interval=1, entry_threshold=50, exit_threshold=-50
):
    price = prices[interval]
    aroonosc = AROONOSC.run(price["high"], price["low"], timeperiods)
    # Run strategy signal generator
    aroon_signals = AroonStrategy.run(aroonosc.real, entry_threshold, exit_threshold)
    return aroon_signals


def generate_aroons(
    prices, timeperiods, interval=1, entry_threshold=50, exit_threshold=-50, **kwargs
):
    price = prices[interval]
    aroons = AROON.run(price["high"], price["low"], timeperiods)
    extreme_up = aroons.aroonup_above(99.99999)
    extreme_down = aroons.aroondown_above(99.99999)

    def replace_extremes(is_up=True):
        def applier(series):
            period = series.name
            extreme_times = series.index[series.values == True]
            if is_up:
                extreme = price.rolling(window=pd.Timedelta(minutes=period * interval))[
                    "high"
                ].max()
            else:
                extreme = price.rolling(window=pd.Timedelta(minutes=period * interval))[
                    "low"
                ].min()
            extreme_prices = extreme.loc[extreme_times]
            df = series.to_frame().join(extreme_prices)
            return df["high" if is_up else "low"]  # .astype(
            #     pd.SparseDtype("float"), np.nan
            # )  # .fillna(0)

        return applier

    extreme_up = extreme_up.apply(replace_extremes(is_up=True))
    extreme_down = extreme_down.apply(replace_extremes(is_up=False))

    aroonosc = AROONOSC.run(price["high"], price["low"], timeperiods)

    entries1 = (
        aroonosc.real_above([e - 0.000001 for e in entry_threshold], crossover=True)
        .astype(int)
        .replace(0, np.nan)
    )
    entries0 = (
        aroonosc.real_below([e - 0.000001 for e in entry_threshold], crossover=True)
        .astype(int)
        .replace(0, np.nan)
        .replace(1, 0)
    )

    entries1.columns = entries1.columns.set_levels(entry_threshold, level=0)
    entries0.columns = entries0.columns.set_levels(entry_threshold, level=0)

    entries = entries1.fillna(entries0)

    exits1 = (
        aroonosc.real_below([e + 0.000001 for e in exit_threshold], crossover=True)
        .astype(int)
        .replace(0, np.nan)
    )
    exits0 = (
        aroonosc.real_above([e + 0.000001 for e in exit_threshold], crossover=True)
        .astype(int)
        .replace(0, np.nan)
        .replace(1, 0)
    )
    exits1.columns = exits1.columns.set_levels(exit_threshold, level=0)
    exits0.columns = exits0.columns.set_levels(exit_threshold, level=0)
    exits = exits1.fillna(exits0)

    def set_extreme_for_threshold(extreme):
        def applier(series):
            threshold, period = series.name
            crossing_point = series.index[series.values == 1]
            relevant_extremes = extreme.loc[crossing_point][period]
            ones = series.loc[series == 1]
            series.rename("ones&zeroes", inplace=True)
            df = (
                series.to_frame()
                .join(relevant_extremes)
                .join(price["open"].loc[ones.index])
            )
            res = (
                df[period]
                .fillna(df["open"])
                .fillna(series)
                .astype(pd.SparseDtype("float", np.nan))
                # .replace(np.nan, -2)
                # .replace(0, np.nan)
                # .replace(-2, 0)
            )
            return res

        return applier

    entries = entries.apply(set_extreme_for_threshold(extreme_up))
    exits = exits.apply(set_extreme_for_threshold(extreme_down))
    return entries, exits, aroonosc.real


def save_parquet(dataframe, filepath=None, writer=None):
    """Method writes/append dataframes in parquet format.

    This method is used to write pandas DataFrame as pyarrow Table in parquet format. If the methods is invoked
    with writer, it appends dataframe to the already written pyarrow table.

    :param dataframe: pd.DataFrame to be written in parquet format.
    :param filepath: target file location for parquet file.
    :param writer: ParquetWriter object to write pyarrow tables in parquet format.
    :return: ParquetWriter object. This can be passed in the subsequenct method calls to append DataFrame
        in the pyarrow Table
    """
    table = pa.Table.from_pandas(dataframe)
    if writer is None:
        writer = pq.ParquetWriter(f"{filepath}.parq", table.schema)
    writer.write_table(table=table)
    return writer


def aWholeNewMain(pair, price_path, startDate, endDate, batches):
    printd("Loading Prices")
    prices = load_multiple_prices(pair, price_path, startDate, endDate)

    output_dir = (
        pathlib.Path(__file__)
        .parent.absolute()
        .joinpath("..", "..", "output", pair, "signals")
    )
    dbpath = output_dir / f"Aroon {current_time}.db"
    printd(f"Connecting to Database at {dbpath}!")
    conn = sqlite3.connect(database=dbpath)
    with tqdm(total=len(batches)) as pbar:
        for idx, batch in enumerate(batches):
            pbar.write(f"[{now()}] {batch=}")
            ts = time()
            pbar.write(f"[{now()}] Generating Aroons")
            entries, exits, aroonosc = generate_aroons(prices, **batch)
            # print(entries, aroonosc)
            entries_sparse = entries.astype(pd.SparseDtype("float", np.nan))
            entries_sparse = entries_sparse.sparse.to_coo()
            # aroonosc = aroonosc.melt(
            #     ignore_index=False, var_name="timeperiod", value_name="oscillator"
            # )
            sparse_entries = pd.DataFrame(
                {
                    "timestamp": entries.index[entries_sparse.row],
                    "timeperiod": entries.columns[entries_sparse.col].get_level_values(
                        1
                    )
                    * batch["interval"],
                    "threshold": entries.columns[entries_sparse.col].get_level_values(
                        0
                    ),
                    "trigger": entries_sparse.data,
                }
            )
            # sparse_entries = sparse_entries.merge(
            #     aroonosc,
            #     how="left",
            #     left_on=["timestamp", "timeperiod"],
            #     right_on=["time", "timeperiod"],
            # )
            pbar.write(
                f"[{now()}] {sparse_entries.memory_usage(index=True).sum() / 1024**2:.2f}MB"
            )
            save_db(conn, sparse_entries, "entries", pbar=pbar)

            exits_sparse = exits.astype(pd.SparseDtype("float", np.nan))
            exits_sparse = exits_sparse.sparse.to_coo()
            sparse_exits = pd.DataFrame(
                {
                    "timestamp": exits.index[exits_sparse.row],
                    "timeperiod": exits.columns[exits_sparse.col].get_level_values(1)
                    * batch["interval"],
                    "threshold": exits.columns[exits_sparse.col].get_level_values(0),
                    "trigger": exits_sparse.data,
                }
            )
            save_db(conn, sparse_exits, "exits", pbar=pbar)

            te = time()
            # printd("Collect Garbage")
            # gcts = time()
            # gc.collect()
            # gcte = time()

            pbar.write(f"[{now()}] Batch {idx} Time taken: {te - ts}")
            pbar.update(1)
    printd("Creating Indexes")
    conn.executescript(
        """
        CREATE INDEX entries_timestamp_idx ON entries (timestamp);
        CREATE INDEX exits_timestamp_idx ON exits (timestamp);
    """
    )

    printd("Converting entries to Parquet")
    db_to_parquet(conn, "entries", output_dir=output_dir)
    printd("Converting exits to Parquet")
    db_to_parquet(conn, "exits", output_dir=output_dir)

    printd("Closing db connection")
    conn.close()


def db_to_parquet(conn, table, output_dir, filename_prefix="aroon_"):
    df = pd.read_sql(
        f"""
        SELECT timestamp, timeperiod, threshold, trigger
        FROM {table}
        ORDER BY timestamp
        """,
        conn,
        parse_dates="timestamp",
    )
    df["timeperiod"] = df["timeperiod"].astype("uint32")
    df["threshold"] = df["threshold"].astype("int8")

    set_aroon_ids(df, table, output_dir)
    # df.to_parquet(
    #     output_dir / f"{current_time} - {filename_prefix}{table}.parquet",
    #     index=False,
    # )


def reidentify_aroons(pair):
    output_dir = (
        pathlib.Path(__file__)
        .parent.absolute()
        .joinpath("..", "..", "output", "signals", pair)
    )
    entries = pq.read_table(
        output_dir / "XBTUSD_aroon_entries_2021-05-18T0058.parquet"
    ).to_pandas()
    set_aroon_ids(entries, "entries", output_dir)

    exits = pq.read_table(
        output_dir / "XBTUSD_aroon_exits_2021-05-18T0058.parquet"
    ).to_pandas()
    set_aroon_ids(exits, "exits", output_dir)


def set_aroon_ids(df: pd.DataFrame, name, output_dir):
    group = df.groupby(["timeperiod", "threshold"])
    ids = pd.DataFrame(group.groups.keys(), columns=["timeperiod", "threshold"])
    ids["id"] = ids.index + 1
    ids.to_parquet(
        output_dir / f"{current_time} - aroon_{name}.ids.parquet",
    )
    df = df.merge(ids, on=["timeperiod", "threshold"], how="left")
    df["id"] = df["id"].astype("uint16")
    df = df.drop(["timeperiod", "threshold"], axis=1)
    df.to_parquet(
        output_dir / f"{current_time} - aroon_{name}.parquet",
    )


@timeme
def main(
    pair,
    path,
    startDate,
    endDate,
    batches,
    save_method=None,
):
    printd("Loading Prices")
    prices = load_multiple_prices(pair, path, startDate, endDate)

    output_dir = (
        pathlib.Path(__file__)
        .parent.absolute()
        .joinpath("..", "..", "output", "signals")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    printd("Starting Batches")
    for batch in batches:
        printd(f"{batch=}")
        ts = time()
        printd("Generating Signals")
        signals = generate_signals(prices, **batch)
        printd("Generating Sparse Entry Matrix")
        entries = sp.sparse.coo_matrix(signals.entries.values)
        sparse_entries = pd.DataFrame(
            {
                "timestamp": signals.entries.index[entries.row],
                "interval": batch["interval"],
                "timeperiod": signals.entries.columns[entries.col].get_level_values(
                    "aroonosc_timeperiod"
                ),
                "threshold": signals.entries.columns[entries.col].get_level_values(
                    "custom_entry_threshold"
                ),
            }
        )
        printd("Generating Sparse Exit Matrix")
        exits = sp.sparse.coo_matrix(signals.exits.values)
        sparse_exits = pd.DataFrame(
            {
                "timestamp": signals.exits.index[exits.row],
                "interval": batch["interval"],
                "timeperiod": signals.exits.columns[exits.col].get_level_values(
                    "aroonosc_timeperiod"
                ),
                "threshold": signals.entries.columns[exits.col].get_level_values(
                    "custom_exit_threshold"
                ),
            }
        )
        if save_method is not None:
            printd(f"Saving...")
            filename = f"{pair} {name}"
            save_method(
                sparse_entries,
                "aroon_entries",
                output_dir.joinpath(filename),
            )
            save_method(
                sparse_exits,
                "aroon_exits",
                output_dir.joinpath(filename),
            )
        te = time()
        printd("Collect Garbage")
        gcts = time()
        gc.collect()
        gcte = time()

        printd(f"Time taken: {te - ts}, GC: {gcte-gcts}")


bitches = []
batch_size = 3

intervals = [1, 5, 15, 60, 720, 1440]

for (i, interval) in enumerate(intervals):
    if interval == 1:
        start = 2
    else:
        start = math.floor((intervals[i - 1] * 60) / interval) + 1
    end = 60

    for batch_start in range(start, end, batch_size):
        bitches.append(
            {
                "interval": interval,
                "timeperiods": list(range(batch_start, batch_start + batch_size)),
                "entry_threshold": list(range(0, 101, 10)),
                "exit_threshold": list(range(0, -101, -10)),
            }
        )


def chunks(array, n):
    """Yield successive n-sized chunks from array."""
    for i in range(0, len(array), n):
        yield array[i : i + n]


funky_periods = (
    list(range(65, math.ceil(365 / 2), 5))
    + list(range(math.ceil(365 / 2), 365, 15))
    + list(range(365, 365 * 3, 30))
)
for chunk in chunks(funky_periods, batch_size):
    bitches.append(
        {
            "interval": 1440,
            "timeperiods": chunk,
            "entry_threshold": list(range(0, 101, 10)),
            "exit_threshold": list(range(0, -101, -10)),
        }
    )
# 1-5 60, 15-60 900 | 720-1440 43200

adj_timeperiods = []
for bitch in bitches:
    adj_timeperiods += [tp * bitch["interval"] for tp in bitch["timeperiods"]]
assert len(adj_timeperiods) == len(set(adj_timeperiods))

pair = "XBTUSD"
path = "../data/kraken"
startDate = "2017-05-15"
endDate = "2021-06-16"
# startDate = "2018-02-26"
# endDate = "2018-02-28"


aWholeNewMain(
    pair=pair,
    price_path=path,
    startDate=startDate,
    endDate=endDate,
    batches=bitches,
)
# reidentify_aroons(pair)


# def update():
#     main(
#         pair=pair,
#         path=path,
#         startDate=startDate,
#         endDate=endDate,
#         interval=interval,
#         skip_save=False,
#         save_method=save_db_update,
#         batches=bitches,
#     )
