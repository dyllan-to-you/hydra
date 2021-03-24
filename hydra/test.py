# import pathlib
# import pyarrow.parquet as pq

# filepath = pathlib.Path(__file__).parent.absolute()
# parqpath = filepath.joinpath('../data/kraken', "XBTUSD_720.filled.parq")
# table = pq.read_table(parqpath)
# prices = table.to_pandas()
# print(prices)

from pathlib import Path
import vectorbt as vbt
from hydra.SuperSim import load_prices

def load_portfolio(name):
    output_dir = Path.cwd().joinpath('..', 'output')
    filepath = output_dir.joinpath(name)
    return vbt.Portfolio.load(filepath)

vbt.settings.ohlcv["column_names"] = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}

pair='XBTUSD'
start_data='2018-05-15'
end_data='2018-07-15'
interval=1

# portfolio = load_portfolio(f"{pair} {start_data} {end_data} {interval} Aroon 100-101.portfolio")
prices = {}
prices[1] = load_prices(pair, '../data/kraken', start_data, end_data, 1)
prices[5] = load_prices(pair, '../data/kraken', start_data, end_data, 5)
prices[15] = load_prices(pair, '../data/kraken', start_data, end_data, 15)
prices[60] = load_prices(pair, '../data/kraken', start_data, end_data, 60)
prices[720] = load_prices(pair, '../data/kraken', start_data, end_data,720)
prices[1440] = load_prices(pair, '../data/kraken', start_data, end_data, 1440)