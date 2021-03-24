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
    res = re.search('.*_(\d+)\.parq',name)
    target = f"{stem}.filled.parq"
    if res is None or dir.joinpath(target).is_file():
        end = timer()
        return f"Skipped {name} {round(end-start, 2)}s" # skip if already filled
    # print('Processing', name)
    interval = res.group(1)
    table = pq.read_table(file)
    prices = table.to_pandas()
    prices["time"] = pd.to_datetime(prices["time"], unit="s")
    prices = prices.set_index("time").asfreq(f"{interval}Min")
    last_close = None
    for idx, row in prices.iterrows():
        if pd.isnull(row["close"]):
            prices.at[idx, 'open'] = last_close
            prices.at[idx, 'high'] = last_close
            prices.at[idx, 'low'] = last_close
            prices.at[idx, 'close'] = last_close
            prices.at[idx, 'volume'] = 0
            prices.at[idx, 'trades'] = 0

        last_close = row['close']

    prices.to_parquet(dir.joinpath(target))
    end = timer()
    return f"Filled {name} {round(end-start, 2)}s"

def main():
    print(str(sys.argv[1]))

    files = glob.glob(str(sys.argv[1]))
    with tqdm(total=len(files)) as pbar:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(process, file): file for file in files}
            for future in concurrent.futures.as_completed(futures):
                pbar.set_description(future.result())
                pbar.update(1)

if __name__ == '__main__':
    main()