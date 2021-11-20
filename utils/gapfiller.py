import math
import glob
import sys
import re
import pathlib
import pandas as pd
from tqdm import tqdm
import pyarrow.parquet as pq
import concurrent.futures
from timeit import default_timer as timer


def process(file):
    start = timer()
    path = pathlib.Path(file).resolve()
    name = path.name
    stem = path.stem
    dir = path.parent
    res = re.search(".*_(\d+)\.parq", name)
    target = f"{stem}.filled.parq"
    if res is None or dir.joinpath(target).is_file():
        end = timer()
        return f"Skipped {name} {round(end-start, 2)}s"  # skip if already filled
    # print('Processing', name)
    interval = res.group(1)
    table = pq.read_table(file)
    prices = table.to_pandas()
    prices = fillGaps(interval, prices)

    prices.to_parquet(dir.joinpath(target))
    end = timer()
    return f"Filled {name} {round(end-start, 2)}s"


def fillGaps(interval, prices: pd.DataFrame, last_close=None, start_time=None):
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices = prices.set_index("time")

    if start_time is None:
        start_time = prices.index[0]
    new_index = pd.date_range(start_time, prices.index[-1], freq=f"{interval}Min")
    prices = prices.reindex(new_index, fill_value=None)

    testIntervalReindex(interval, prices)
    for idx, row in prices.iterrows():
        if pd.isna(row["close"]):
            print("found null")
            prices.at[idx, "open"] = last_close
            prices.at[idx, "high"] = last_close
            prices.at[idx, "low"] = last_close
            prices.at[idx, "close"] = last_close
            prices.at[idx, "volume"] = 0
            prices.at[idx, "trades"] = 0

        last_close = row["close"]
    return prices


def testIntervalReindex(interval, prices):
    intervalTest = pd.date_range(
        prices.index[0], prices.index[-1], freq=f"{interval}Min"
    ).tolist()
    assert len(intervalTest) == len(
        prices.index
    ), f"""
        Target length: {len(intervalTest)}, Actual length: {len(prices.index)}
    """


def main():
    print(str(sys.argv[1]))

    files = glob.glob(str(sys.argv[1]))
    with tqdm(total=len(files)) as pbar:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(process, file): file for file in files}
            for future in concurrent.futures.as_completed(futures):
                pbar.set_description(future.result())
                pbar.update(1)


if __name__ == "__main__":
    main()