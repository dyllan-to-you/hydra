import pathlib
import pandas as pd
import pyarrow.parquet as pq

price_cache = {}


def load_prices(
    pair, path="../data/kraken", startDate=None, endDate=None, interval=1
) -> pd.DataFrame:
    filename = f"{pair}_{interval}"
    prices = price_cache.get(filename, None)
    if prices is None:
        filepath = pathlib.Path(__file__).parent.absolute()
        parqpath = filepath.joinpath(path, f"{filename}.filled.parq")
        table = pq.read_table(parqpath)
        prices = table.to_pandas()
        price_cache[filename] = prices
    # print(prices.index)
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices["time"] = pd.to_datetime(prices["time"], unit="s")
        prices = prices.set_index("time")
    if startDate is not None and endDate is not None:
        prices = prices.loc[startDate:endDate]
    return prices.drop("trades", axis=1)


def get_simulation_id(id_base, entry, exit):
    return entry * id_base + exit