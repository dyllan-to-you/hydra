from datetime import datetime
import os
import pathlib
from configparser import ConfigParser
import pandas as pd
import numpy as np
from binance import Client
from dataloader import pairs, kraken
from hydra.utils import timeme
from utilsPkg.gapfiller import fillGaps

FILEPATH = pathlib.Path(__file__).parent.absolute()


def load_prices(
    pair,
    startDate=None,
    endDate=None,
    interval=1,
    log=False,
):
    b_interval = f"{interval}m"
    datadir = FILEPATH.joinpath("../data/binance.us", f"{pair}_{b_interval}")

    if log:
        print("LOAD LOG", datadir)
        print(pair, startDate, endDate, b_interval)
    if startDate is not None and endDate is not None:
        start: datetime = pd.to_datetime(startDate)
        startYear = start.year
        startMonth = start.month
        startDay = start.day
        end = pd.to_datetime(endDate)
        endYear = end.year
        endMonth = end.month
        endDay = end.day
        if log:
            print(startYear, startMonth, startDay, endYear, endMonth, endDay)

        if startYear == endYear:
            if startMonth == endMonth:
                filters = [
                    [
                        ("year", "==", startYear),
                        ("month", "==", startMonth),
                        ("day", ">=", startDay),
                    ]
                ]
            else:
                filters = [
                    [
                        ("year", "==", startYear),
                        ("month", "==", startMonth),
                        ("day", ">=", startDay),
                    ],
                    [
                        ("year", "==", startYear),
                        ("month", ">", startMonth),
                        ("month", "<", endMonth),
                    ],
                    [
                        ("year", "==", endYear),
                        ("month", "==", endMonth),
                        ("day", "<=", endDay),
                    ],
                ]
        else:
            filters = [
                [
                    ("year", "==", startYear),
                    ("month", "==", startMonth),
                    ("day", ">=", startDay),
                ],
                [
                    ("year", "==", startYear),
                    ("month", ">", startMonth),
                ],
                [
                    ("year", ">", startYear),
                    ("year", "<", endYear),
                ],
                [
                    ("year", "==", endYear),
                    ("month", "<", endMonth),
                ],
                [
                    ("year", "==", endYear),
                    ("month", "==", endMonth),
                    ("day", "<=", endDay),
                ],
            ]

        prices = pd.read_parquet(
            datadir,
            filters=filters,
        )
        if log:
            print("prices a", prices)
        prices = prices.loc[(prices.index >= start) & (prices.index <= end)]
        if log:
            print("prices b", prices)
    elif startDate is not None:
        Exception("No endtime is not supported yet")
        start: datetime = pd.to_datetime(startDate)
        startYear = start.year
        startMonth = start.month
        startDay = start.day
        filters = [
            [
                ("year", "==", startYear),
                ("month", "==", startMonth),
                ("day", ">=", startDay),
            ],
            [
                ("year", "==", startYear),
                ("month", ">", startMonth),
            ],
            [
                ("year", ">", startYear),
            ],
        ]

        prices = pd.read_parquet(datadir, filters=filters)
        prices = prices.loc[(prices.index >= start)]
    else:
        prices = pd.read_parquet(
            datadir,
        )
    prices = prices.drop(["year", "month", "day"], axis=1, errors="ignore")

    _interval = f"{interval}T"
    prices = prices.groupby(pd.Grouper(freq=_interval)).agg(
        {"open": "first", "close": "last", "high": "max", "low": "min", "volume": "sum"}
    )
    # prices = prices.iloc[:-1]
    prices = prices.sort_index()
    return prices


@timeme
def download(
    client: Client,
    pair,
    start="2000/01/01",
    end=None,
    interval=1,
):
    b_interval = f"{interval}m"

    if end is not None:
        klines = client.get_historical_klines_generator(pair, b_interval, start, end)
    else:
        klines = client.get_historical_klines_generator(pair, b_interval, start)
    klines_df = pd.DataFrame(
        klines,
        columns=[
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "Close time",
            "Quote asset volume",
            "Number of trades",
            "Taker buy base asset volume",
            "Taker buy quote asset volume",
            "ignore",
        ],
    )
    klines_df["time"] = pd.to_datetime(klines_df["time"], unit="ms")
    klines_df = klines_df.drop(
        [
            "Close time",
            "Quote asset volume",
            "Number of trades",
            "Taker buy base asset volume",
            "Taker buy quote asset volume",
            "ignore",
        ],
        axis=1,
    )
    klines_df = klines_df.set_index("time")
    klines_df["open"] = pd.to_numeric(klines_df["open"])
    klines_df["high"] = pd.to_numeric(klines_df["high"])
    klines_df["low"] = pd.to_numeric(klines_df["low"])
    klines_df["close"] = pd.to_numeric(klines_df["close"])
    klines_df["volume"] = pd.to_numeric(klines_df["volume"])

    return klines_df


def set_partition_keys(df):
    df["year"] = df.index.year.astype(int)
    df["month"] = df.index.month.astype(int)
    df["day"] = df.index.day.astype(int)
    return df


def partition_filename_cb(keys):
    year, month, day = keys
    return f"{int(year)}-{int(month)}-{int(day)}.parq"


@timeme
def update_data(pair_binance="BTCUSD", interval=1):
    b_interval = f"{interval}m"
    pair_kraken = pairs.binance_to_kraken(pair_binance)

    config = ConfigParser()

    config.read(FILEPATH.joinpath("../configs/dyllan.ini"))

    client = Client(
        config.get("main", "BINANCE_APIKEY"),
        config.get("main", "BINANCE_SECRET"),
        tld="us",
    )

    datadir = FILEPATH.joinpath("../data/binance.us", f"{pair_binance}_{b_interval}")
    datadir.mkdir(parents=True, exist_ok=True)
    print(datadir)
    files = list(datadir.glob(f"**/*.parq"))
    if len(files) == 0:
        if pair_kraken is not None:
            kraken_prices = kraken.load_prices(pair_kraken)
            lastDT = kraken_prices.index[-1]
            target_start = lastDT + pd.to_timedelta(f"{interval}min")
            lastClose = kraken_prices["close"][-1]
            print(f"{lastClose=}, {lastDT=}, {target_start=}")
        else:
            target_start = "2000/01/01"
            lastClose = None
        data = download(client, pair_binance, str(target_start))
        data = fillGaps(
            interval,
            data,
            last_close=lastClose,
            start_time=target_start,
        )
        data = set_partition_keys(data)
        data.to_parquet(
            datadir,
            index=True,
            partition_cols=["year", "month", "day"],
            partition_filename_cb=partition_filename_cb,
        )
    else:
        lastFile = sorted(files, key=lambda x: pd.to_datetime(x.stem))[-1]
        print(lastFile.name)
        lastData = pd.read_parquet(lastFile)
        lastClose = lastData["close"][-1]
        lastDT = pd.to_datetime(lastData.index[-1])

        target_start = lastDT + pd.to_timedelta(f"{interval}min")
        data = download(client, pair_binance, start=str(target_start))
        data = fillGaps(
            interval,
            data,
            last_close=lastClose,
            start_time=target_start,
        )
        # Note: to_parquet will overwrite the files for relevant partitions, thus the concat
        data = set_partition_keys(pd.concat([lastData, data]))

        data.to_parquet(
            datadir,
            index=True,
            partition_cols=["year", "month", "day"],
            partition_filename_cb=partition_filename_cb,
        )
    # print(data)


if __name__ == "__main__":
    # update_data(pair_binance="DOGEUSD", pair_kraken=None)
    update_data(pair_binance="BTCUSD")
    # load_prices("BTCUSD")
