import pathlib
import pandas as pd
import pyarrow.parquet as pq


def load_prices(pair, path, startDate=None, endDate=None, interval=1) -> pd.DataFrame:
    filepath = pathlib.Path(__file__).parent.absolute()
    parqpath = filepath.joinpath(path, f"{pair}_{interval}.filled.parq")
    table = pq.read_table(parqpath)
    prices = table.to_pandas()
    # print(prices.index)
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices["time"] = pd.to_datetime(prices["time"], unit="s")
        prices = prices.set_index("time")
    if startDate is not None and endDate is not None:
        prices = prices.loc[startDate:endDate]
    return prices.drop("trades", axis=1)


def get_simulation_id(id_base, entry, exit):
    return entry * id_base + exit