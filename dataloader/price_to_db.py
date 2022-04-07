import re
import pathlib
import duckdb
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from hydra.utils import timeme, printd

FILEPATH = pathlib.Path(__file__).parent.absolute()


def getExistingTickers(con, table):
    con.execute(f"SELECT DISTINCT ticker FROM {table}")
    return [ticker for (ticker, *_) in con.fetchall()]


def saveKraken(con):
    path = FILEPATH / "../data/kraken"
    filelist = list(path.rglob(f"*_1.filled.parq"))

    # Check for table existence
    try:
        con.execute("PRAGMA table_info('kraken');")
        print(con.fetchall())
    # Create table if not exists
    except:
        con.execute(
            f"""CREATE TABLE kraken(
                ticker VARCHAR,
                "timestamp" TIMESTAMP, 
                open DOUBLE, 
                high DOUBLE, 
                low DOUBLE, 
                "close" DOUBLE, 
                volume DOUBLE, 
                PRIMARY KEY (ticker, "timestamp")
            );"""
        )

    existingTickers = getExistingTickers(con, "kraken")
    print("existing tickers:", existingTickers)

    pbar = tqdm(filelist)
    for file in pbar:
        pattern = "(\w+)_1\.filled\.parq"
        match = re.search(pattern, str(file))
        ticker = match.group(1)
        pbar.set_description(f"Kraken Ticker: {ticker}")
        if ticker in existingTickers:
            pbar.write(f"Ticker {ticker} exists")
            continue

        table = pq.read_table(file)
        prices = table.to_pandas()
        prices.index = prices.index.rename("timestamp")
        prices["ticker"] = ticker
        prices = prices.reset_index()
        viewname = f"kraken_{ticker}_view"
        con.register(viewname, prices)
        con.execute(
            f"""INSERT INTO kraken 
        SELECT ticker, timestamp, open, high, low, close, volume 
        FROM {viewname}"""
        )
        pbar.write(f"Inserted Ticker {ticker} size:{con.fetchall()[0][0]}")
        existingTickers.append(ticker)

    pbar.close()

    tickerCheck = getExistingTickers(con, "binance")
    print(tickerCheck)
    # assert set(existingTickers) == set(
    #     tickerCheck
    # ), f"{set(existingTickers) - set(tickerCheck)} | {set(tickerCheck) - set(existingTickers)}"


def saveBinance(con):
    ticker = "BTCUSD"
    datadir = FILEPATH.joinpath("../data/binance.us", f"BTCUSD_1m")
    prices = pd.read_parquet(
        datadir,
    )

    try:
        con.execute("PRAGMA table_info('binance');")
        print(con.fetchall())
    except:
        con.execute(
            f"""CREATE TABLE binance(
                ticker VARCHAR,
                "timestamp" TIMESTAMP, 
                open DOUBLE, 
                high DOUBLE, 
                low DOUBLE, 
                "close" DOUBLE, 
                volume DOUBLE, 
                PRIMARY KEY (ticker, "timestamp")
            );"""
        )

    prices["ticker"] = ticker
    prices = prices.drop(columns=["trades", "year", "month", "day"])
    prices.index = prices.index.rename("timestamp")
    prices = prices.reset_index()
    # prices.index = prices.index.rename("timestamp")
    print(prices)
    viewname = f"binance_{ticker}_view"
    con.register(viewname, prices)

    con.execute(
        f"""INSERT INTO binance 
        SELECT ticker, timestamp, open, high, low, close, volume 
        FROM {viewname}"""
    )
    con.fetchall()


@timeme
def query(con, query):
    con.execute(query)
    res = con.fetchall()
    print(len(res))


# NOTE: Does not work with nodejs
# @timeme
# def stdb():
#     printd("connecting")
#     con = duckdb.connect(
#         database="../data/prices.duckdb",
#     )
#     printd("connected")

#     # saveKraken(con)
#     # saveBinance(con)

#     # print(len(getExistingTickers(con, "kraken")))
#     # con.execute(
#     #     "SELECT * from kraken WHERE ticker='XBTUSD' AND timestamp > '2019-01-01' and timestamp < '2019-01-02'"
#     # )
#     # res = con.fetchall()
#     # print(len(res))

#     query(
#         con,
#         "SELECT * from kraken WHERE ticker='XBTUSD' AND timestamp > '2019-01-01' and timestamp < '2019-01-02'",
#     )
#     query(
#         con,
#         "SELECT * from kraken WHERE ticker='XBTUSD' AND timestamp > '2019-06-05' and timestamp < '2019-06-07'",
#     )
#     query(
#         con,
#         "SELECT * from kraken WHERE ticker='XBTUSD' AND timestamp > '2019-03-01' and timestamp < '2019-03-01 12:00'",
#     )
#     query(
#         con,
#         "SELECT * from binance WHERE ticker='XBTUSD' AND timestamp > '2020-06-01' and timestamp < '2020-06-03 12:00'",
#     )

#     printd("closing")
#     con.close()
#     printd("closed")


@timeme
def pqdb():
    con = duckdb.connect(
        # database="../data/prices.duckdb",
    )

    dbexportPath = "../data/prices.duckpq"
    try:
        con.execute(f"IMPORT DATABASE '{dbexportPath}'")
    except:
        pass

    # saveBinance(con)
    # saveKraken(con)

    # print(len(getExistingTickers(con, "kraken")))
    con.execute(
        "SELECT * from kraken WHERE ticker='XBTUSD' AND timestamp > '2019-01-01' and timestamp < '2019-01-02'"
    )
    res = con.fetchall()
    print(len(res))

    # print(f"Exporting database as parquet to {dbexportPath}")
    # con.execute(f"EXPORT DATABASE '{dbexportPath}' (FORMAT PARQUET);")
    con.close()


# pqdb()
stdb()