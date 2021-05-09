from tqdm.std import tqdm
from hydra.PriceLoader import load_prices
from hydra.LivinOnAPrayer import Direction, get_simulation_id
from math import floor
from hydra.utils import now, printd, timeme, write
import sqlite3
import pyarrow as pa
import pyarrow.parquet as pq
import pprint
import pandas as pd

# from distributed import Client
# client = Client(memory_limit="8G")
# import modin.pandas as pd

pp = pprint.PrettyPrinter(indent=2)
MAX_MEMORY_MB = 4 * 1024
last_entry = 46
last_exit = 12


def gen_parq(conn, table):
    df = pd.read_sql(
        f"SELECT id, timestamp from {table}",
        conn,
        parse_dates="timestamp",
    )

    # df["timestamp"] = df["timestamp"].astype("int64") // 10 ** 9
    # df["timestamp"] = df["timestamp"].astype("uint32")
    df["id"] = df["id"].astype("uint16")
    printd(df.dtypes, df.index)
    df.to_parquet(f"{table}.ts.parquet")


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


@timeme
def main(*args, last_entry=0, last_exit=0):
    entries_df = pd.read_parquet("aroon_entry.parquet")
    exits_df = pd.read_parquet("aroon_exit.parquet")

    entries_df["timestamp"] = pd.to_datetime(entries_df["timestamp"], unit="s")
    exits_df["timestamp"] = pd.to_datetime(exits_df["timestamp"], unit="s")

    entries_df["val"] = 1
    exits_df["val"] = -1

    entry_ids = entries_df["id"].unique()[:]
    exit_ids = exits_df["id"].unique()[:]

    prices = load_prices("XBTUSD", "../data/kraken")[["open"]]
    prices["open"] = prices["open"].astype("float32")
    print(prices[0:5])
    id_base = max(len(entry_ids), len(exit_ids))
    saves = 0
    acc = []
    acc_mem = 0
    with tqdm(total=(len(entry_ids) - last_entry) * len(exit_ids)) as pbar:
        for entry_idx, entry in enumerate(tqdm(entry_ids)):
            if entry_idx < last_entry:
                pbar.set_description(f"Skipping Entries {entry_idx}/{last_entry}")
                continue
            entries = entries_df.loc[entries_df["id"] == entry]
            entries = entries.set_index("timestamp").drop("id", axis=1)
            for exit_idx, exit in enumerate(tqdm(exit_ids, leave=False)):
                if entry_idx == last_entry and exit_idx <= last_exit:
                    pbar.update(1)
                    pbar.set_description(
                        f"Skipping Exits {exit_idx}/{last_exit} for entry {last_entry}"
                    )
                    continue
                simulation_id = get_simulation_id(id_base, entry, exit)
                exits = exits_df.loc[exits_df["id"] == exit]
                exits = exits.set_index("timestamp").drop("id", axis=1)
                orders = exits.join(entries, how="outer", lsuffix="ex")
                order = orders.val.fillna(0) + orders.valex.fillna(0)
                buys_idx = crossings_nonzero_neg2pos(order.to_numpy())
                sells_idx = crossings_nonzero_pos2neg(order.to_numpy())

                buys = prices.loc[order.iloc[buys_idx].index]
                sells = prices.loc[order.iloc[sells_idx].index]

                buys["direction"] = True
                sells["direction"] = False
                order_prices = pd.concat([buys, sells]).sort_index(kind="merge")[
                    ["open", "direction"]
                ]
                order_prices["year"] = order_prices.index.year.astype("uint16")
                order_prices["month"] = order_prices.index.month.astype("uint8")
                order_prices["simulation"] = simulation_id.astype("uint32")
                acc.append(order_prices)
                acc_mem += order_prices.memory_usage(index=True).sum() / 1024 / 1024

                pbar.set_description(
                    f"{saves=} {len(acc)=} mem={acc_mem:.2f}/{MAX_MEMORY_MB}MB"
                )
                if acc_mem >= MAX_MEMORY_MB:
                    pbar.set_description(
                        f"{saves=} {len(acc)=} mem={acc_mem:.2f}/{MAX_MEMORY_MB}MB [SAVING]"
                    )
                    pq.write_to_dataset(
                        pa.Table.from_pandas(
                            pd.concat(acc),
                            columns=[
                                "simulation",
                                "open",
                                "direction",
                                "year",
                                "month",
                            ],
                        ),
                        root_path="F:\\hydra\\orders",
                        partition_cols=["year", "month"],
                        compression="brotli",
                        compression_level=6,
                    )
                    acc_mem = 0
                    acc = []
                    saves += 1
                    write(
                        f"Entry: {entry_idx=} Exit: {exit_idx=} {simulation_id}",
                        "Saved Orders.txt",
                    )
                    last_entry = entry_idx
                    last_exit = exit_idx
                pbar.update(1)


if __name__ == "__main__":
    try:
        conn = sqlite3.connect(
            database="output/signals/XBTUSD Aroon 2021-04-16T2204.db"
        )
        # cur = conn.cursor()
        main(conn, last_entry=last_entry, last_exit=last_exit)
        # gen_parq(conn, "aroon_entry")
        # gen_parq(conn, "aroon_exit")
    except KeyboardInterrupt as e:
        print("KeyboardInterrupt", e)
    finally:
        conn.close()
    # ray.shutdown()
