import sqlite3
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


current_time = datetime.now().strftime("%Y-%m-%dT%H%M")


def save_db(df, table, filepath, *args, **kwargs):
    printd(f"Connecting to Database!")
    conn = sqlite3.connect(database=f"{filepath} {current_time}.db")
    printd(f"Converting dataframe to table")
    df.to_sql(f"{table}_df", conn, if_exists="replace", index=False)
    try:
        db_stmnt = f"INSERT INTO {table} SELECT * FROM {table}_df;"
        printd(f"{db_stmnt=}")
        conn.execute(db_stmnt)
    except sqlite3.OperationalError as e:
        if "no such table:" not in str(e):
            print(e)
            raise e
        db_stmnt = f"""
            CREATE TABLE {table} AS SELECT * FROM {table}_df;
            CREATE INDEX {table}_timestamp ON {table} (timestamp);
        """
        printd(f"{db_stmnt=}")
        conn.executescript(db_stmnt)

    conn.commit()
    printd("Closing db connection")
    conn.close()


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
    printd("Loading Prices")
    prices = load_prices(pair, path, startDate, endDate, interval)
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
        filename = f"{pair} {name}"
        printd(f"Generating Signals")
        signals = generate_signals(prices, **batch)
        printd(f"Generating Sparse Entry Matrix")
        entries = sp.sparse.coo_matrix(signals.entries.values)
        sparse_entries = pd.DataFrame(
            {
                "timestamp": signals.entries.index[entries.row],
                "timeperiod": signals.entries.columns[entries.col],
            }
        )
        printd(f"Generating Sparse Exit Matrix")
        exits = sp.sparse.coo_matrix(signals.exits.values)
        sparse_exits = pd.DataFrame(
            {
                "timestamp": signals.exits.index[exits.row],
                "timeperiod": signals.exits.columns[exits.col],
            }
        )
        if not skip_save and save_method is not None:
            printd(f"Saving...")
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
start = 2
batch_size = 20
for i in range(0, 10000):
    bitches.append(
        {
            "timeperiods": list(
                range(start + (batch_size * i), start + (batch_size * (i + 1)))
            )
        }
    )


pair = "XBTUSD"
path = "../data/kraken"
startDate = "2017-05-15"
endDate = "2017-06-16"
interval = 1


def null():
    main(
        pair=pair,
        path=path,
        startDate=startDate,
        endDate=endDate,
        interval=interval,
        skip_save=True,
        batches=bitches,
    )


def save():
    printd("Running SAVE")
    main(
        pair=pair,
        path=path,
        startDate=startDate,
        endDate=endDate,
        interval=interval,
        skip_save=False,
        save_method=save_db,
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
#         save_method=save_db_update,
#         batches=bitches,
#     )
