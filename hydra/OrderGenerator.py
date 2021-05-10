import gc
from timeit import default_timer as timer
from tqdm.std import tqdm
from hydra.SimManager import load_prices, get_simulation_id
from math import floor
from hydra.utils import get_mem, now, printd, timeme, write
import sqlite3
import pyarrow as pa
import pyarrow.parquet as pq
import pprint

import pandas as pd

# from distributed import Client

# if __name__ == "__main__":
#     client = Client(memory_limit="8G")
# import modin.pandas as pd

pp = pprint.PrettyPrinter(indent=2)
MAX_MEMORY = 4 * 1024 ** 3
MAX_MEMORY_MB = MAX_MEMORY / 1024 ** 2
last_entry = 2
last_exit = 115


@timeme
def gen_parq(conn, table):
    df = pd.read_sql(
        f"SELECT id, timestamp from {table}",
        conn,
        parse_dates="timestamp",
    )

    df["id"] = df["id"].astype("uint16")
    df.to_parquet(f"{table}.ts.parquet", index=False)


# @timeme
# def dask_main(db=None):
#     entries = dd.read_parquet("aroon_entry.parquet")

#     entries_sparse = sparse.COO([entries.iloc[:, 0], entries.iloc[:, 1]], 1)
#     entries_da = da.from_array(entries_sparse, asarray=False)
#     entries_da.compute()
# exits = dd.read_parquet("aroon_exit.parquet")
# df.set_index("id")
# printd(entries_df)

# printd(entries_da[:5, :5])


def crossings_nonzero_pos2neg(data):
    pos = data > 0
    return (pos[:-1] & ~pos[1:]).nonzero()[0] + 1


def crossings_nonzero_neg2pos(data):
    pos = data > 0
    return (~pos[:-1] & pos[1:]).nonzero()[0] + 1


def partition_filename(prefix="", suffix=""):
    def fn(keys):
        return f"{prefix}{'-'.join([str(k) for k in keys])}{suffix}.parquet"

    return fn


@timeme
def main(last_entry=None, last_exit=None, skip_save=False):
    if last_entry is None:
        last_entry = 0
    entries_df = pd.read_parquet("aroon_entry.parquet")
    exits_df = pd.read_parquet("aroon_exit.parquet")

    entries_df["timestamp"] = pd.to_datetime(entries_df["timestamp"], unit="s")
    exits_df["timestamp"] = pd.to_datetime(exits_df["timestamp"], unit="s")
    entries_df = entries_df.assign(val=1)
    exits_df = exits_df.assign(val=-1)
    entries_df = entries_df.set_index(["id", "timestamp"])
    exits_df = exits_df.set_index(["id", "timestamp"])
    # print(entries_df, exits_df)

    entry_ids = entries_df.index.unique(level=0)
    exit_ids = exits_df.index.unique(level=0)

    prices = load_prices("XBTUSD", "../data/kraken")[["open"]]
    prices["open"] = prices["open"].astype("float32")
    id_base = max(len(entry_ids), len(exit_ids))
    saves = 0
    acc = {"buys": [], "sells": [], "mem": 0}
    gc.collect()
    ts = timer()
    with tqdm(total=(len(entry_ids) - last_entry) * len(exit_ids)) as pbar:
        for entry_idx, (entry_id, entries) in enumerate(
            tqdm(entries_df.groupby(level=0))
        ):
            if entry_idx < last_entry:
                pbar.set_description(f"Skipping Entries {entry_idx}/{last_entry-1}")
                continue
            entries = entries.reset_index(level=0, drop=True)
            # entries = entries_df.loc[entry, :]
            # entries = entries_df.loc[entries_df["id"] == entry]
            # entries = entries.set_index("timestamp").drop("id", axis=1)
            for exit_idx, (exit_id, exits) in enumerate(
                tqdm(exits_df.groupby(level=0), leave=False)
            ):
                if (
                    last_exit is not None
                    and entry_idx == last_entry
                    and exit_idx <= last_exit
                ):
                    pbar.update(1)
                    pbar.set_description(
                        f"Skipping Exits {exit_idx}/{last_exit} for entry {last_entry}"
                    )
                    continue
                if len(acc["buys"]) % 25 == 0:
                    pbar.set_description(
                        f"""{saves=} {len(acc["buys"])=} mem={acc["mem"]/1024**2:.2f}/{
                            MAX_MEMORY_MB}"""
                    )

                simulation_id = get_simulation_id(id_base, entry_id, exit_id)
                exits = exits.reset_index(level=0, drop=True)
                # exits = exits_df.loc[exit, :]
                # exits = exits_df.loc[exits_df["id"] == exit]
                # exits = exits.set_index("timestamp").drop("id", axis=1)
                orders = exits.join(entries, how="outer", lsuffix="ex")
                orders = orders.val.fillna(0) + orders.valex.fillna(0)
                buys_idx = crossings_nonzero_neg2pos(orders.to_numpy())
                sells_idx = crossings_nonzero_pos2neg(orders.to_numpy())

                if len(buys_idx) == 0 or len(sells_idx) == 0:
                    continue
                buys = prices.loc[orders.iloc[buys_idx].index]
                sells = prices.loc[orders.iloc[sells_idx].index]
                try:
                    if buys.index[0] > sells.index[0]:
                        sells = sells.iloc[1:]
                    if len(sells) == 0:
                        continue
                    if buys.index[-1] > sells.index[-1]:
                        buys = buys.iloc[:-1]
                    if len(buys) == 0:
                        continue
                    if len(buys) != len(sells):
                        raise Exception("ohgodwhyplease")
                except Exception as e:
                    pbar.write(f"O Buy: {prices.loc[orders.iloc[buys_idx].index]}")
                    pbar.write(f"O Sell: {prices.loc[orders.iloc[sells_idx].index]}")
                    pbar.write(f"Buy {buys}")
                    pbar.write(f"Sells {sells}")
                    raise e

                buys = buys.reset_index()
                sells = sells.reset_index()

                trades = buys.join(sells, lsuffix="_buy", rsuffix="_sell")
                sells["profit_open"] = trades["open_sell"] / trades["open_buy"]

                buys["year"] = buys["timestamp"].dt.year.astype("uint16")
                buys["month"] = buys["timestamp"].dt.month.astype("uint8")
                buys["day"] = buys["timestamp"].dt.day.astype("uint8")
                buys["simulation"] = simulation_id
                buys["simulation"] = buys["simulation"].astype("uint32")
                buys = buys.drop(columns=["open"])

                sells["year"] = sells["timestamp"].dt.year.astype("uint16")
                sells["month"] = sells["timestamp"].dt.month.astype("uint8")
                sells["day"] = sells["timestamp"].dt.day.astype("uint8")
                sells["simulation"] = simulation_id
                sells["simulation"] = sells["simulation"].astype("uint32")
                sells = sells.drop(columns=["open"])

                acc["buys"].append(buys)
                acc["sells"].append(sells)
                acc["mem"] += (
                    buys.memory_usage(index=True).sum()
                    + sells.memory_usage(index=True).sum()
                )
                # acc["mem"] = get_mem()

                if acc["mem"] >= MAX_MEMORY:
                    pbar.set_description(
                        f"{saves=} {len(acc['buys'])=} "
                        f"mem={acc['mem'] / 1024**2:.2f}/{MAX_MEMORY_MB}MB [SAVING]"
                    )
                    save_start = timer()
                    if not skip_save:
                        # Buys
                        pq.write_to_dataset(
                            pa.Table.from_pandas(
                                pd.concat(acc["buys"]),
                                columns=[
                                    "timestamp",
                                    "simulation",
                                    "year",
                                    "month",
                                    "day",
                                ],
                                preserve_index=False,
                            ),
                            root_path="F:/hydra/orders",
                            partition_cols=["year", "month", "day"],
                            partition_filename_cb=partition_filename(suffix=".buys"),
                            compression="brotli",
                            compression_level=6,
                        )

                        # Sells
                        pq.write_to_dataset(
                            pa.Table.from_pandas(
                                pd.concat(acc["sells"]),
                                columns=[
                                    "timestamp",
                                    "simulation",
                                    "profit_open",
                                    "year",
                                    "month",
                                    "day",
                                ],
                            ),
                            root_path="F:/hydra/orders",
                            partition_cols=["year", "month", "day"],
                            partition_filename_cb=partition_filename(suffix=".sells"),
                            compression="brotli",
                            compression_level=6,
                        )
                    save_stop = timer()
                    save_time = save_stop - save_start

                    te = timer()
                    elapsed = te - ts
                    ts = timer()

                    write(
                        f"{entry_idx=} {exit_idx=} {simulation_id=}",
                        "Saved Orders.txt",
                    )
                    pbar.write(
                        f"[{now()}] {entry_idx=} {exit_idx=} {simulation_id=} "
                        f"SaveTime={save_time:2.4f}s Elapsed={elapsed:2.4f}s "
                        f"{len(acc['buys'])=}"
                    )

                    last_entry = entry_idx
                    last_exit = exit_idx
                    acc = {"buys": [], "sells": [], "mem": 0}
                    saves += 1
                pbar.update(1)


if __name__ == "__main__":
    try:
        # conn = sqlite3.connect(
        #     database="output/signals/XBTUSD Aroon 2021-04-16T2204.db"
        # )
        # cur = conn.cursor()
        # gen_parq(conn, "aroon_entry")
        # gen_parq(conn, "aroon_exit")
        main(last_entry=last_entry, last_exit=last_exit)
    except KeyboardInterrupt as e:
        print("KeyboardInterrupt", e)
    # finally:
    #     conn.close()
