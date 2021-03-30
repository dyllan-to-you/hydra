import duckdb
import pathlib
import numpy as np
from numba import njit
import pandas as pd
import vectorbt as vbt
from vectorbt.signals.factory import SignalFactory
from hydra.SuperSim import load_prices
import pyarrow as pa
import pyarrow.parquet as pq
from hydra.utils import timeme, now, printd
import gc
from time import time
from datetime import datetime
import os, psutil
import scipy as sp

process = psutil.Process(os.getpid())

name = "Aroon"


@njit
def aroon_entry(from_i, to_i, col, a, temp_idx_arr):
    if from_i == 0:
        w = np.where(a[:, col] > 50)[0]
        for i, num in enumerate(w):
            temp_idx_arr[i] = num

    for i in range(len(temp_idx_arr)):
        if temp_idx_arr[i] > from_i and temp_idx_arr[i] < to_i:
            return temp_idx_arr[i : i + 1]
    return temp_idx_arr[:0]


@njit
def aroon_exit(from_i, to_i, col, a, temp_idx_arr):
    if temp_idx_arr[-1] != 42:
        temp_idx_arr[-1] = 42
        w = np.where(a[:, col] < -50)[0]
        for i, num in enumerate(w):
            temp_idx_arr[i] = num

    for i in range(len(temp_idx_arr)):
        if temp_idx_arr[i] > from_i and temp_idx_arr[i] < to_i:
            return temp_idx_arr[i : i + 1]
    return temp_idx_arr[:0]


AroonStrategy = SignalFactory(input_names=["aroon"]).from_choice_func(
    entry_choice_func=aroon_entry,
    entry_settings=dict(
        pass_inputs=["aroon"],
        pass_kwargs=["temp_idx_arr"],  # built-in kwarg
    ),
    exit_choice_func=aroon_exit,
    exit_settings=dict(
        pass_inputs=["aroon"],
        pass_kwargs=["temp_idx_arr"],  # built-in kwarg
    ),
    # forward_flex_2d=True,
)

AROONOSC = vbt.IndicatorFactory.from_talib("AROONOSC")
# printd(help(AROONOSC.run))


def generate_signals(prices, timeperiods):
    aroonosc = AROONOSC.run(prices["high"], prices["low"], timeperiods)
    # Run strategy signal generator
    aroon_signals = AroonStrategy.run(aroonosc.real)
    return aroon_signals


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


current_time = datetime.now().strftime("%Y%m%dT%H%M%S")


def save_duck(df, table, filepath, *args, **kwargs):
    cursor = duckdb.connect(database=f"{filepath}.{current_time}.duck")
    try:
        cursor.register(f"{table}_view", df)
        cursor.execute(f"INSERT INTO {table} SELECT * FROM {table}_view")

        # cursor.execute("""DESCRIBE aroon_the_world;""")
        # print("DESCRIBE join", cursor.fetchall())
    except RuntimeError as e:
        if "Catalog Error: Table with name" not in str(e):
            print(e)
            raise e
        # df.insert(0, "timestamp", df.index)
        cursor.register(f"{table}_view", df)
        cursor.execute(f"CREATE TABLE {table} AS SELECT * FROM {table}_view; ")
        cursor.execute(f"CREATE INDEX {table}_timestamp ON {table} (timestamp);")
    cursor.close()


@timeme
def main(
    pair,
    path,
    startDate,
    endDate,
    interval,
    batches,
    save_method=None,
    skip_save=False,
):
    prices = load_prices(pair, path, startDate, endDate, interval)
    output_dir = (
        pathlib.Path(__file__)
        .parent.absolute()
        .joinpath("..", "..", "output", "signals")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for batch in batches:
        ts = time()
        filename = f"{pair} {name}"
        signals = generate_signals(prices, **batch)
        entries = sp.sparse.coo_matrix(signals.entries.values)
        sparse_entries = pd.DataFrame(
            {
                "timestamp": signals.entries.index[entries.row],
                "timeperiod": signals.entries.columns[entries.col],
            }
        )

        exits = sp.sparse.coo_matrix(signals.exits.values)
        sparse_exits = pd.DataFrame(
            {
                "timestamp": signals.exits.index[exits.row],
                "timeperiod": signals.exits.columns[exits.col],
            }
        )
        if not skip_save and save_method is not None:
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
        timeperiods = batch["timeperiods"]
        printd(
            f"{te - ts}, {timeperiods[0]}-{timeperiods[-1]}, {process.memory_info().rss / 1024}"
        )


# main(
#     pair="XBTUSD",
#     path="../data/kraken",
#     startDate="2017-05-15",
#     endDate="2021-05-16",
#     interval=1,
#     skip_save=True,
#     batches=[
#         {
#             "timeperiods": list(range(100, 102)),
#         },
#         {
#             "timeperiods": list(range(102, 104)),
#         },
#     ],
# )

bitches = []
for i in range(0, 10000):
    bitches.append({"timeperiods": list(range(2 + (20 * i), 100 + (20 * (i + 1))))})


pair = "XBTUSD"
path = "../data/kraken"
startDate = "2018-05-15"
endDate = "2021-05-16"
interval = 1


def null():
    main(
        pair=pair,
        path=path,
        startDate=startDate,
        endDate=endDate,
        interval=interval,
        skip_save=True,
        save_method=save_duck_join,
        batches=bitches,
    )


def save():
    main(
        pair=pair,
        path=path,
        startDate=startDate,
        endDate=endDate,
        interval=interval,
        skip_save=False,
        save_method=save_duck,
        batches=bitches,
    )


# def update():
#     main(
#         pair=pair,
#         path=path,
#         startDate=startDate,
#         endDate=endDate,
#         interval=interval,
#         skip_save=False,
#         save_method=save_duck_update,
#         batches=bitches,
#     )
