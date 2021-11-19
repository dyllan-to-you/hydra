from dataloader.binance_data import update_data
from dataloader import load_prices
from datetime import datetime
from hydra.Environments import main as bulk_analysis, fft_price_analysis
import pathlib
import pandas as pd
import pickle

"""
Read and Write a "WIndowTracker" file. 
WindowTracker has window and last run columns
On run, read WindowTracker, create new "Environment" tasks for each window that needs to be rerun
When running a window, all subwindows are removed and only replaced if the relevant wavelength is still significant
If Root Window needs to be rerun, erase all subwindows and start Environment task for that window
    Create Environment Tasks for all significant frequency/wavelengths, recursively
    Update WindowTracker file
Wavelength > Year > Month.csv > run row
"""


def main(pair_binance="BTCUSD", overlap=0.99, forceNewFile=True, config="config"):
    update_data(pair_binance=pair_binance)
    output_dir = pathlib.Path("./output/enviro")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_file = output_dir / config

    try:
        with open(config_file, "rb") as handle:
            file = pickle.load(handle)
            lastRun = file["lastRun"]
            startDate = (lastRun - pd.to_timedelta("365d")).floor("D")
    except FileNotFoundError:
        prices = load_prices(b_pair="BTCUSD", k_pair=("XBTUSD"))
        startDate = (prices.index[0] + pd.to_timedelta("1439min")).floor("D")
        # lastRun = startDate + pd.to_timedelta("365d")

    results = bulk_analysis(
        startDate, window="365d", detrend=True, pair=pair_binance, midnightLock=True
    )
    print(results)
    return

    # endDate = (startDate + pd.to_timedelta("365d")).floor("D")
    # startDate = endDate - pd.to_timedelta("365d")
    # with open(config_file, "wb") as handle:
    #     pickle.dump(file, handle)

    # windowTracker_path = output_dir / "windowTracker.csv"
    # try:
    #     if forceNewFile:
    #         raise FileNotFoundError()
    #     windowTracker = pd.read_csv(windowTracker_path)  # .readText()
    # except FileNotFoundError:
    #     windowTracker = pd.DataFrame([{"window": "365d", "lastRun": "2021-11-01"}])

    # now = datetime.now()
    # for window, lastRun in windowTracker.itertuples(index=False, name=None):
    #     window_delta = pd.to_timedelta(window)
    #     lastRun_date = pd.to_datetime(lastRun)

    #     if window == "365d":
    #         overlap_delta = pd.DateOffset("1d")

    #     else:
    #         overlap_delta = window_delta * (1 - overlap)
    #     if lastRun_date + overlap_delta > now:
    #         print("SKIP", lastRun_date + overlap_delta)
    #         continue
    #     else:
    #         startDate = lastRun_date - window_delta + overlap_delta
    #         endDate = lastRun_date + overlap_delta
    #         print(startDate, window_delta)
    #         res = fft_price_analysis(
    #             pair=pair,
    #             startDate=startDate,
    #             endDate=lastRun_date,
    #             detrend=True,
    #             window=window_delta,
    #         )
    #         print(res)
    # windowTracker.to_csv(windowTracker_path, index=False)


if __name__ == "__main__":
    main()