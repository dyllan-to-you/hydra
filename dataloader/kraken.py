import pathlib
from typing import List
import pandas as pd
import pandas.core.groupby as pdGroupby
import pyarrow.parquet as pq

price_cache = {}


def load_prices(pair, startDate=None, endDate=None, interval=1) -> pd.DataFrame:
    path = "../data/kraken"
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
    elif startDate is not None:
        prices = prices.loc[startDate:]
        prices = prices.iloc[:-1]
    return prices.drop("trades", axis=1)


def load_output_signal(output_dir, file) -> pd.DataFrame:
    path = output_dir / f"{file}.parquet"
    return pq.read_table(path).to_pandas()


def load_parquet_by_date(dir, date, name, core=0):
    if isinstance(core, list):
        frames = [
            load_output_signal(
                dir / f"year={date.year}/month={date.month}/core={c}",
                f"day={date.day}.{name}",
            )
            for c in core
        ]

        df = pd.concat(frames, ignore_index=True)
        return df

    return load_output_signal(
        dir / f"year={date.year}/month={date.month}/core={core}",
        f"day={date.day}.{name}",
    )


class ParquetLoader:
    name: str
    location: pathlib.Path
    loaded_date: pd.Timestamp
    loaded_chunks: List[int]
    loaded_df: pd.DataFrame
    cached_file: pdGroupby.GroupBy

    def __init__(self, location, name):
        self.name = name
        self.location = location
        self.loaded_date = None
        self.loaded_chunks = None
        self.loaded_df = None
        self.cached_file = None

    def get_file(self, date: pd.Timestamp, chunks):
        if (
            self.loaded_date is None
            or self.loaded_date.year != date.year
            or self.loaded_date.dayofyear != date.dayofyear
            or self.loaded_chunks != chunks
        ):
            self.cached_file = load_parquet_by_date(
                self.location, date, self.name, chunks
            )
            # print(self.cached_file)
            self.loaded_chunks = chunks
            self.cached_file = self.cached_file.iloc[
                self.cached_file["timestamp"].searchsorted(date, side="left") :
            ]
            self.cached_file = self.cached_file.groupby("timestamp").__iter__()
            self.loaded_date, self.loaded_df = next(self.cached_file, (None, None))
        else:
            while self.loaded_date is not None and self.loaded_date < date:
                self.loaded_date, self.loaded_df = next(self.cached_file, (None, None))

        return self.loaded_df if self.loaded_date == date else None