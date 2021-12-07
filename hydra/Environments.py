# matplotlib.use("TkAgg")
import os
import pathlib
from pprint import pprint
import sys
import traceback
import pickle
import wave
from ray.exceptions import RayTaskError

from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import ray
from scipy import stats
import seaborn as sns

from dataloader import load_prices
from hydra.utils import printd, timeme


# pd.set_option("display.max_rows", 50)
# pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 0)
sns.set_theme()

LOG_RAY = False
NUM_CORES = 60


PROPORTION = 0.025
PRICE_DEVIANCE_CUTOFF = 0.01
PRICE_DEVIANCE_PORTION = 0.01

# 56 @ 99.95
# 165 @ 99.5
# 251 @ 99
# 548 @ 95


@ray.remote
def fft_price_analysis_ray(*args, **kwargs):
    try:
        return fft_price_analysis(*args, **kwargs)
    except Exception as err:
        msg = "".join(traceback.format_exception(type(err), err, err.__traceback__))
        print(msg, args, kwargs)
        raise err


class Trim:
    def __init__(self, trim_len=None):
        self.len = trim_len

    def array(self, arr):
        if self.len is None or self.len == 0:
            return arr
        return arr[self.len : -self.len]


@timeme
def fft_price_analysis(
    pair,
    startDate,
    window=None,
    window_original=None,
    endDate=None,
    detrend=True,
):
    np.seterr(all="raise")

    startDate = pd.to_datetime(startDate)
    if window is not None:
        endDate = startDate + window
    endDate = pd.to_datetime(endDate)
    time_step = 1 / 60 / 24
    fig_dates = (
        f"{startDate.strftime('%Y-%m-%d %H:%M')} - {endDate.strftime('%Y-%m-%d %H:%M')}"
    )
    figname = f"{pair} {fig_dates}"
    prices = load_prices(pair, startDate=startDate, endDate=endDate)["open"]
    assert not np.isnan(prices).any()
    assert len(prices) % 2 == 0  # Even # of prices is faster
    price_len = len(prices)
    printd(f"\n=+=+=+=+=+=+=+=+=+=+= START {figname} =+=+=+=+=+=+=+=+=+=+=")
    # print("PRICES", prices)

    if len(prices) == 0 or endDate.floor("1min") != prices.index[-1]:
        print("ERROR", len(prices), window, startDate, endDate)
        print(prices.index[0], prices.index[-1])

        raise Exception("Help oh god please no")

    trim = Trim(round(price_len * PROPORTION) or None)
    price_trim = trim.array(prices)  # why is this here
    slope = 0
    intercept = np.mean(prices)
    tf = [price for price in prices if not price >= 0]
    assert len(tf) == 0

    if detrend:
        figname = "(D)" + figname
        slope, intercept, *_ = stats.linregress(
            np.array(range(price_len)),
            prices,
        )
    assert not np.isnan(slope).any() and not np.isnan(intercept).any()
    line_gen = date_line_gen_factory(slope, intercept, startDate)
    trendline = line_gen(startDate, endDate, index=True)
    if len(prices) != len(trendline):
        print(prices[-10:], trendline[-10:])
        print(startDate, endDate)
    price_detrended = prices - trendline
    price_detrended_trim = trim.array(price_detrended)
    trendline_trim = trim.array(trendline)

    try:
        assert len(price_detrended_trim)
    except Exception as e:
        print(prices)
        print(trim)
        print(price_detrended_trim)
        raise e

    # print("TREND", trendline, len(trendline), startDate, endDate)
    # print(price_len, len(prices), slope, intercept, trendline, price_detrended)
    valuable_info = transform(price_detrended, time_step)
    fft, freqs, index, powers = valuable_info
    # print(f"{len(fft)=} {fft=}")
    # print(f"{len(freqs)=} {freqs=}")

    # printd("PRICES", prices, prices.shape, key_count)
    df: pd.DataFrame = df_from_ifft_variance(fft, freqs, price_detrended_trim, trim)

    df["minPerCycle"] = 1440 / df["frequency"].to_numpy()

    constructed = construct_price_significant_frequencies(
        df, fft, freqs, price_detrended, trendline, trim
    )
    if constructed is None:
        printd(
            "constructed is none?",
            figname,
            df.size,
            fft.size,
            len(freqs),
        )
        raise Exception("Constructed is None")

    (
        approximated_price,  # Price constructed using the most impactful frequencies
        deviances,
        subset_removed,
        frequencies_kept,
    ) = constructed
    frequencies_kept.name = startDate

    # max_amplitude_ifft=None
    extrapolations = [
        extrapolate_ifft(
            key,
            {
                "frequency": row[0],
                "variance": row[1],
                "amplitude": row[2],
                "amplitude_normalized": row[3],
                "minPerCycle": row[4],
                "deviance": row[5],
            },
            fft,
            prices.index,
            trendline_gen=line_gen,
        )
        for key, *row in frequencies_kept.itertuples()
    ]

    extrapolations_df = pd.DataFrame(
        extrapolations,
        columns=[
            # "ifft_extrapolated",
            # "ifft_extrapolated_trended",
            "minPerCycle",
            "ifft_extrapolated_wavelength",
            "ifft_extrapolated_amplitude",
            "ifft_extrapolated_deviance",
            "first_extrapolated",
            "first_extrapolated_date",
            "first_extrapolated_isup",
        ],
    )
    print("RESULTS ==========================")

    frequency_extrapolations = frequencies_kept.merge(
        extrapolations_df, on="minPerCycle"
    )

    last_price = prices.iloc[-1]

    significant_extrapolations = frequency_extrapolations.loc[
        frequency_extrapolations["amplitude"] >= last_price * 0.005
    ]
    insignificant_extrapolations = frequency_extrapolations.loc[
        frequency_extrapolations["amplitude"] < last_price * 0.005
    ]
    significant_extrapolation_line = significant_extrapolations[
        ["first_extrapolated_date", "first_extrapolated"]
    ].set_index("first_extrapolated_date")["first_extrapolated"]
    insignificant_extrapolation_line = insignificant_extrapolations[
        ["first_extrapolated_date", "first_extrapolated"]
    ].set_index("first_extrapolated_date")["first_extrapolated"]

    # add trendline as freq 0
    trend_step = window_original * 0.01
    trend_step_m = trend_step.total_seconds() / 60
    if trend_step_m < 1:
        trend_step_m = 1
        trend_step = pd.to_timedelta("1min")
    trend_prediction = line_gen(
        endDate,
        endDate + trend_step,
        trend_step_m,
        index=True,
        inclusive=True,
    )

    # add trend_prediction as frequency: 0
    significant_extrapolations = significant_extrapolations.append(
        {
            "frequency": 0,
            "variance": 0,
            "amplitude": 0,
            "amplitude_normalized": 0,
            "minPerCycle": 0,
            "deviance": 0,
            "ifft_extrapolated_wavelength": pd.to_timedelta("0min"),
            "ifft_extrapolated_amplitude": 0,
            "ifft_extrapolated_deviance": 0,
            "first_extrapolated": trend_prediction.values[-1],
            "first_extrapolated_date": trend_prediction.index[-1],
            "first_extrapolated_isup": slope > 0,
        },
        ignore_index=True,
    )

    window = window + pd.to_timedelta(1, unit="min")
    meta = dict(
        figname=figname,
        startDate=startDate,
        endDate=endDate,
        window=window,
        window_original=window_original,
        # significant_wavelengths=frequencies_kept["minPerCycle"],
        trend_deviance=deviance_calc(trendline_trim, 0, price_trim),
        trend_slope=slope,
        trend_intercept=intercept,
        # extrapolated=ifft_extrapolated_trended,
        # extrapolated_wavelength=ifft_extrapolated_wavelength,
        # extrapolated_amplitude=ifft_extrapolated_amplitude,
        # extrapolated_deviance=ifft_extrapolated_deviance,
        # first_extrapolated=first_extrapolated[0],
        # first_extrapolated_date=first_extrapolated[1],
        # first_extrapolated_isup=first_extrapolated[2],
    )
    x = significant_extrapolations["minPerCycle"].apply(lambda x: pd.Series(meta))

    interesting = significant_extrapolations.join(x)
    interesting["figname"] = (
        interesting["figname"] + "|w" + interesting["minPerCycle"].astype(str)
    )
    interesting = interesting.set_index("figname")

    plotty = dict(
        startDate=startDate,
        endDate=endDate,
        figname=figname,
        window=window,
        window_original=window_original,
        fig_dates=fig_dates,
        # subset_removed=subset_removed,
        deviance=deviances,
        # price=prices,
        # price_detrended=price_detrended,
        # trendline=pd.Series(trendline, index=price_index),
        trend_slope=slope,
        trend_intercept=intercept,
        # approximated_price=approximated_price + trendline,
        # detrended_approximated_price=approximated_price,
        # fft=fft,
        frequencies_kept=frequencies_kept,
        # extrapolated=ifft_extrapolated_trended,
        # extrapolated_wavelength=ifft_extrapolated_wavelength,
        # extrapolated_amplitude=ifft_extrapolated_amplitude,
        # first_extrapolated=first_extrapolated,
        significant_extrapolation_line=significant_extrapolation_line,
        insignificant_extrapolation_line=insignificant_extrapolation_line,
    )
    return interesting, plotty


def extrapolate_ifft(key, row, fft: np.ndarray, datetime_index, trendline_gen):

    # print("EXTRAPOLATE ========================")
    # print(row)

    ifft = pd.Series(get_ifft_by_index(fft, key, Trim(0))[0], index=datetime_index)
    wavelength = pd.to_timedelta(row["minPerCycle"], unit="m")
    last_cycle_start = datetime_index[-1] - wavelength
    last_cycle = ifft[last_cycle_start:]
    last_max_idx = last_cycle.idxmax()
    last_min_idx = last_cycle.idxmin()
    # print(last_max_idx, last_min_idx)
    delta = wavelength / 2  # abs(last_max_idx - last_min_idx)

    endpoint = datetime_index[-1] + (datetime_index[-1] - datetime_index[0])

    # print(f"{last_min_idx=} {last_max_idx=} {endpoint=}")
    # trendline = trendline_gen(datetime_index[0], datetime_index[-1], index=True)

    extrapolated_trendline = trendline_gen(
        min(last_min_idx, last_max_idx) + delta * 2,
        endpoint,
        interval=delta.total_seconds() / 60,
        index=True,
        inclusive=True,
    )

    extras = []
    start_pole = 1 if last_max_idx < last_min_idx else -1
    for date in extrapolated_trendline.index:
        extras.append(row["amplitude"] * start_pole)
        start_pole *= -1

    extrapolated = pd.Series(extras, extrapolated_trendline.index)
    extrapolated_trended = extrapolated + extrapolated_trendline
    # print("LAST CYCLE", trendline.loc[[last_max_idx, last_min_idx]])
    return (
        # extrapolated,
        # extrapolated_trended,
        row["minPerCycle"],
        wavelength,
        row["amplitude"],
        row["deviance"],
        extrapolated_trended[0],
        extrapolated_trended.index[0],
        extrapolated[0] > 0,
    )  # pd.concat([ifft, extrapolated])


"""
    extra_up = []
    extra_down = []
    start_pole = 1 if last_max_idx < last_min_idx else -1
    for date in extrapolated_trendline.index:
        val = row["amplitude"] * start_pole
        if val > 0:
            extra_up.append((date, val))
        else:
            extra_down.append((date, val))
        start_pole *= -1

    # extrapolated = pd.Series(extras, extrapolated_trendline.index)
    extrapolated_up = pd.Series(extra_up)
    extrapolated_down = pd.Series(extra_down)
    extrapolated = pd.Series(dict([*extrapolated_down, *extrapolated_up]))
    print("LAST CYCLE", trendline.loc[[last_max_idx, last_min_idx]])
    print("EXTRPOLATD", extrapolated)
    print(
        "up",
        extrapolated_up,
    )
    print("dwn", extrapolated_down)
    return extrapolated, extrapolated_trendline  # pd.concat([ifft, extrapolated])
"""


@timeme
def construct_price_significant_frequencies(
    df: pd.DataFrame,
    fft,
    freqs,
    price_detrended: pd.Series,
    trendline: np.ndarray,
    trim: int,
):
    price_detrended_add = pd.Series(
        np.zeros(len(price_detrended)),
        index=price_detrended.index,
    )
    # print("price", price_detrended)
    deviances = []
    count = 0
    subset_removed = [1]

    df["keep"] = np.zeros(len(df), dtype=bool)
    # print(df)

    empty_fft = None
    for key, frequency, *_ in df.sort_values(["variance"], ascending=False).itertuples(
        name=None
    ):
        count += 1
        ifft, empty_fft = get_ifft_by_index(fft, key, Trim(0), empty_fft)
        price_detrended_add = price_detrended_add + ifft
        deviance = deviance_calc(
            trim.array(price_detrended_add),
            trim.array(trendline),
            trim.array(price_detrended),
        )
        deviances.append(deviance)
        subset_removed.append((df.shape[0] - count) / df.shape[0])
        df.loc[key, "keep"] = True
        if deviance < PRICE_DEVIANCE_CUTOFF:
            keep = df[df["keep"]].drop("keep", axis=1)
            # print(keep)
            # print(len(deviances), len(subset_removed))
            keep["deviance"] = deviances

            return (price_detrended_add, deviances, subset_removed, keep)


def transform(values, time_step):
    freqs = np.fft.fftfreq(values.size, time_step)
    index = np.argsort(freqs)
    fft, powers = get_fft(values)
    return fft, freqs, index, powers


def get_fft(values):
    fft = np.fft.fft(values)
    fft[0] = 0
    powers = fft.real ** 2
    return fft, powers


@timeme
def get_fft_buckets(valuable_info):
    fft, freqs, index, powers = valuable_info

    minute_buckets = {}
    fft_res = pd.DataFrame(dict(power=fft, freqs=freqs))
    # print("fft_res", fft_res)
    # printd("fft,freqs", fft_res)
    for idx, freq in enumerate(freqs):
        min_per_cycle = None
        if freq == 0:
            min_per_cycle = 0
        else:
            min_per_cycle = round(1440 / freq)  # if freq != 0 else 0
            if min_per_cycle == 0:
                min_per_cycle = 1

        # printd(idx, freq, power, min_per_cycle)
        bucket = minute_buckets.setdefault(min_per_cycle, [])
        bucket.append(idx)
    # printd("Keys", sorted([k for k in minute_buckets.keys() if k > 0]))
    minute_bucket_df = pd.DataFrame(
        dict(minutes=minute_buckets.keys(), fft=minute_buckets.values())
    )
    minute_bucket_df["powers"] = [
        np.abs(power) ** 2 for power in minute_bucket_df["fft"]
    ]
    minute_bucket_df["count"] = [len(power) for power in minute_bucket_df["fft"]]
    minute_bucket_df["cum_count"] = minute_bucket_df[["count"]].cumsum()
    # printd(minute_bucket_df)
    return minute_buckets, minute_bucket_df


def get_keys(*arr, keep=False):
    _arr = list(arr)
    # if keep:
    #     _arr += [0]
    return _arr + [x * -1 for x in _arr]


def get_ifft_by_key(fft, minute_bucket, key, price_trim_index, trim):
    # printd("GET IFFT BY KEY", key)
    # [zero] = minute_bucket[0]
    # printd("ZERO", zero)
    keepers = get_keys(key, keep=True)
    keep_me = np.concatenate([minute_bucket[key] for key in keepers])
    strip_idx = np.isin(fft, keep_me, assume_unique=True, invert=True)
    keeped = np.copy(fft)
    keeped[strip_idx] = 0
    ifft = np.fft.ifft(keeped)
    # zeroed = np.fft.ifft(np.where(keeped == zero, 0, keeped))

    ifft_trim = ifft.trim.array(real)
    ifft_series = pd.Series(ifft_trim, index=price_trim_index)

    return (key, ifft_series, len(keep_me))


def get_ifft_by_index(fft, idx, trim, empty_fft=None):
    if empty_fft is None:
        empty_fft = np.zeros(len(fft), dtype=np.complex128)
    empty_fft[idx] = fft[idx]
    empty_fft[-idx] = fft[-idx]
    ifft = np.fft.ifft(empty_fft)
    empty_fft[idx] = 0
    empty_fft[-idx] = 0
    ifft_trim = trim.array(ifft)

    imaginary_booty = np.any(ifft_trim.imag > 0.01)
    if imaginary_booty:
        raise Exception("Booty too big")
        # printd(f"============ WARNING BOOTY {idx} TOO BIG ============")
        # printd(ifft_trim[imaginary_booty])

    return ifft_trim.real, empty_fft


@timeme
def df_from_ifft_variance(fft, freqs, price_detrended_trim: pd.Series, trim: Trim):
    return pd.DataFrame(
        iterate_ifft_variance_np(fft, freqs, price_detrended_trim, trim),
        columns=["key", "frequency", "variance", "amplitude", "amplitude_normalized"],
    ).set_index("key")


def iterate_ifft_variance_np(fft, freq, price_detrended_trim: pd.Series, trim: Trim):
    mid = len(fft) // 2
    empty_fft = None
    fft_amplitude = (
        np.sqrt((np.abs(fft.real) / len(fft)) ** 2 + (np.abs(fft.imag) / len(fft)) ** 2)
        * 2  # Since the power is split between positive and negative freq, multiply by 2
    )
    fft_amplitude_sum = np.sum(fft_amplitude)
    if fft_amplitude_sum == 0:
        fft_amplitude_normalized = fft_amplitude
    else:
        fft_amplitude_normalized = fft_amplitude / fft_amplitude_sum
    for idx in range(1, mid):
        ifft_trim, empty_fft = get_ifft_by_index(fft, idx, trim, empty_fft)
        """
        Note: This returns a matrix of the shape
        array([ [1.        , 0.37405599],
                [0.37405599, 1.        ]])
        """
        x = price_detrended_trim.values
        y = ifft_trim
        try:
            r_value = np.corrcoef(x, y)[0, 1]
            variance = r_value ** 2
            yield idx, freq[idx], variance, fft_amplitude[
                idx
            ], fft_amplitude_normalized[idx]
        except Exception as e:
            # Sometimes x and y can be all zeroes
            if len(x) == 0 or len(y) == 0:
                print("+++ spooky scary skeletrons")
                print(x, price_detrended_trim)
                print(y)

            if np.all(x == x[0]) or np.all(y == y[0]):
                yield idx, freq[idx], 0, fft_amplitude[idx], fft_amplitude_normalized[
                    idx
                ]
                continue
            print("CorrCoeff divide by zero error")
            print("Frequency", mid)
            print(x)
            print(y)
            raise e


def date_line_gen_factory(slope, intercept, originalStartDate, *args, **kwargs):
    def date_line_gen(start, end, interval=1, index=False, inclusive=True):
        # interval = int(interval)
        startDate = pd.to_datetime(start)
        startDelta = startDate - originalStartDate
        startDelta_m = startDelta.total_seconds() / 60

        endDate = pd.to_datetime(end)
        delta = endDate - originalStartDate

        delta_m = delta.total_seconds() / 60
        # Round to interval to resolve "end boundary" issues
        # np.arange excludes the `end` range, however if `end` is a decimal then it is not excluded correctly
        delta_m = interval * np.floor(delta_m / interval)

        line = line_generator(
            slope,
            intercept,
            start=startDelta_m,
            stop=delta_m,
            interval=interval,
            inclusive=inclusive,
        )
        if not index:
            return line
        dt = pd.to_timedelta(f"{interval}min")
        idx = pd.date_range(startDate, endDate, freq=dt)

        try:
            if isinstance(interval, float) and len(idx) - 1 == len(line):
                idx = idx[:-1]
            elif isinstance(interval, float) and len(idx) == len(line) - 1:
                line = line[:-1]
            return pd.Series(line, idx)
        except Exception as e:
            print("Line Gen index mismatch")
            print(len(line), line)
            print(len(idx), idx)
            print(slope, intercept, originalStartDate)
            print(startDate, endDate, startDelta_m, delta_m, interval, inclusive)
            raise e

    return date_line_gen


def line_generator(
    slope, intercept, stop, start=None, interval=1, inclusive=True
) -> np.ndarray:
    if inclusive:
        stop = stop + 1
    if start is None:
        return np.arange(stop, step=interval) * slope + intercept
    else:
        return np.arange(start, stop, step=interval) * slope + intercept


def deviance_calc(constructed_sum, trendline, price_detrended):
    constructed_price = constructed_sum + trendline
    price = price_detrended + trendline
    constructed_and_price_diff = np.abs(constructed_price - price) / price
    sorted_diff = np.flip(np.sort(constructed_and_price_diff))
    sorted_diff_len = len(sorted_diff)
    sorted_diff_prop = int(np.ceil(sorted_diff_len * PRICE_DEVIANCE_PORTION)) or 1
    sorted_diff_subset = sorted_diff[0:sorted_diff_prop]
    try:
        deviance = np.mean(sorted_diff_subset)
        return deviance
    except Exception as e:
        print("==============")
        print(
            f"{price_detrended.index[0]}-{price_detrended.index[-1]} {sorted_diff_prop=} {sorted_diff_subset=}"
        )
        print(e)
        raise e


@timeme
def render_charts(
    figname,
    fig_dates,
    subset_removed,
    deviance,
    price,
    price_detrended,
    trendline,
    approximated_price,
    fft,
    frequencies_kept,
    **kwargs,
):
    fig, axs = plt.subplots(2, num=figname)
    fig.suptitle(fig_dates)
    axs[0].set_title("subset deviance")
    axs[0].plot(subset_removed, deviance, ".")
    render_price_chart(
        price,
        price_chart=[
            # ("detrended", price_detrended),
            ("approximated", approximated_price),
            ("trendline", trendline),
        ],
        # price_chart=[
        #     (
        #         "summed",
        #         cutoffs[-1]["constructed_sum"],
        #         cutoffs[-1]["constructed_coeff"],
        #     ),
        #     *cutoffs[-1]["subset"].itertuples(index=False, name=None),
        # ],
        # price_offset=trendline,
        figname=fig_dates,
        ax=axs[1],
    )


def render_price_chart(
    prices,
    price_chart=[],
    price_offset=0,
    figname="",
    ax=None,
):
    if ax is None:
        ax = plt.gca
    ax.set_title(f"{figname} Prices")
    # plt.plot(prices.index, signal.detrend(prices))
    ax.plot(prices.index, prices, label="price")
    for label, p, *_ in price_chart:
        # coeff = prices.corr(pd.Series(p, index=prices.index))
        # printd(label, p, coeff)
        ax.plot(prices.index, p + price_offset, label=f"{label} ({_})")


def bucket_frequencies(frequencies) -> pd.DataFrame:
    df = frequencies.copy()
    # print(df)
    # df["minPerCycle"] = frequencies.index  # np.round(1440 / frequencies.index.to_numpy())
    df = df.iloc[::-1]
    # df["diff"] = df["minPerCycle"].diff()
    # df["diff%"] = df["diff"].div(df["minPerCycle"].shift(1))
    # df["cumdiff%"] = df["diff%"].cumsum()
    # df["cumdiff%mod"] = df["cumdiff%"].div(0.1)
    # df["cumdiff%moddiff"] = df["cumdiff%mod"].sub(df["cumdiff%mod"].shift(1))
    # df["line"] = 0
    # df.loc[df["cumdiff%moddiff"] >= 1, "line"] = 1
    # print("FREQ", df)
    buckets = {}
    agg = []
    start = None
    last = None
    acc = 0
    for minutes, group in df.groupby(df.index):
        agg.append(group)
        if last is None:
            start = last = minutes
            continue
        if start is None:
            start = minutes
        acc += (minutes - last) / last

        if acc > 0.1:
            label = f"{round(start,3)}-{round(minutes,3)}"
            catted = pd.concat(agg)
            catted["label"] = label
            buckets[start] = catted
            acc = 0
            agg = []
            start = None
            last = minutes
            # print(label, catted, catted.shape)
            continue
        last = minutes
    bucketed: pd.DataFrame = pd.concat(buckets)
    # bucketed = bucketed.drop(columns=["minPerCycle"])
    label_map = bucketed[["label"]]
    label_map = label_map.droplevel(1).drop_duplicates()
    # print(label_map)
    # print(bucketed)
    bucketed = bucketed.groupby(level=0).sum().join(label_map).set_index("label")
    # print(bucketed)
    return bucketed


def process_aggregate(data, chart):
    # print("PROCESS AGGREGATE =======")

    frequencies_kept_df = pd.concat(
        [
            pd.Series(
                #
                data=d["frequencies_kept"]["amplitude_normalized"].values,
                index=d["frequencies_kept"]["frequency"],
                name=d["startDate"],
            )
            for d in chart
        ],
        axis=1,
    )
    # print(frequencies_kept_df)
    frequencies_kept_df = frequencies_kept_df.replace(0, np.nan)
    first_idx = frequencies_kept_df.first_valid_index()
    last_idx = frequencies_kept_df.last_valid_index()
    frequencies_kept_df = frequencies_kept_df.loc[first_idx:last_idx]

    frequencies_kept_df = bucket_frequencies(frequencies_kept_df)
    maxes = frequencies_kept_df.max(axis=1)
    frequencies_kept_df = frequencies_kept_df.divide(maxes, axis=0)

    return dict(
        values=pd.concat([d["frequencies_kept"] for d in chart], axis=1),
        normalized=frequencies_kept_df,
    )


def render_agg_chart(frequencies_kept_df):
    fig, axs = plt.subplots(1, num="Aggregate")
    fig.suptitle("Aggregate")
    # print(
    #     "aggregate",
    # )
    # print(frequencies_kept_df)
    # print(frequencies_kept_df.index)
    # width = frequencies_kept_df.index[1] - frequencies_kept_df.index[0]
    # axs.bar(frequencies_kept_df.index, frequencies_kept_df["sum"], width=width)
    current_cmap = matplotlib.cm.get_cmap("afmhot_r").copy()
    current_cmap.set_bad(color="black")
    # afmhot_r
    sns.heatmap(frequencies_kept_df, cmap=current_cmap, ax=axs, robust=True)
    axs.tick_params(axis="x", labelrotation=-90)


@timeme
def run_parallel(tasks, keep_ray_running=False):
    if sys.argv[-1] == "ray":
        errorThrown = False
        try:
            if not ray.is_initialized():
                ray.init(
                    include_dashboard=True, local_mode=False, log_to_driver=LOG_RAY
                )
            # cProfile.run("main()")

            result_refs = []
            for idx, task in enumerate(tasks):
                if len(result_refs) > NUM_CORES:
                    ray.wait(result_refs, num_returns=idx - NUM_CORES)
                if isinstance(task, tuple):
                    result_refs.append(fft_price_analysis_ray.remote(*task))
                else:
                    result_refs.append(
                        fft_price_analysis_ray.remote(*task["args"], **task["kwargs"])
                    )
            return ray.get(result_refs)
        except KeyboardInterrupt as e:
            errorThrown = True
            printd("Interrupted")
            raise e
        except (RayTaskError, Exception) as e:
            errorThrown = True
            traceback.print_tb(e.__traceback__)
            printd(f"{e}")
            raise e
            # return None
        finally:
            if errorThrown or (not keep_ray_running and ray.is_initialized()):
                ray.shutdown()
    else:
        return [
            fft_price_analysis(*task)
            if isinstance(task, tuple)
            else fft_price_analysis(*task["args"], **task["kwargs"])
            for task in tasks
        ]


def gen_tasks(
    start,
    window: pd.Timedelta,
    pair,
    midnightLock=False,
    end=None,
    overlap=None,
    detrend=True,
    tuple=False,
):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    # Even numbered price length can result in a 3x speedup!
    window_original = window
    if (window.floor("1min") % pd.to_timedelta("2min")).total_seconds() == 0:
        window = window - pd.to_timedelta(1, unit="min")
        if midnightLock:
            start += pd.to_timedelta("1min")

    if midnightLock:
        window_increment = pd.DateOffset(days=1)
    else:
        window_increment = window
        if overlap is not None:
            window_increment *= 1 - overlap
            window_increment = window_increment.round("1min")
        # noninclusive of first run to avoid duplicates when running recursively
        start += window_increment
    if window_increment == pd.to_timedelta("0min"):
        window_increment = pd.to_timedelta("1min")

    prices = load_prices(pair, startDate=start, endDate=end)["open"]
    start = prices.index[0]
    end = prices.index[-1]

    try:
        for startDate in pd.date_range(start=start, end=end, freq=window_increment):
            if startDate + window > end:
                break
            if midnightLock:
                assert (startDate + window).minute == 0
                # startDate = startDate.floor("D")
            if tuple:
                yield (pair, startDate, window, window_original)
            else:
                yield {
                    "args": (pair, startDate),
                    "kwargs": {
                        "detrend": detrend,
                        "window": window,
                        "window_original": window_original,
                    },
                }
    except ZeroDivisionError as e:
        print("Task Gen Divide by Zero Error", start, end, window_increment)
        raise e


def main(
    start,
    window,
    detrend,
    pair,
    overlap=None,
    end=None,
    midnightLock=False,
):
    start = pd.to_datetime(start)
    window_delta = pd.to_timedelta(window)

    tasks = list(
        gen_tasks(
            start,
            end=end,
            window=window_delta,
            overlap=overlap,
            detrend=detrend,
            pair=pair,
            midnightLock=midnightLock,
        )
    )
    results = run_parallel(tasks)
    if results is None:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    results = [result for result in results if result is not None]
    data, charts = zip(*results)
    # print("Tasks:", pd.DataFrame(tasks))
    # print("Results:", len(results))

    aggregates = process_aggregate(data, charts)
    return data, charts, aggregates


if __name__ == "__main__":
    #  start="2020-01-01",
    # end="2021-11-10",
    start = "2018-01-01"
    window = "7d"
    end = str(pd.to_datetime(start) + pd.to_timedelta(window) * 5)
    inputs = dict(
        start=start,
        end=end,
        window=window,
        overlap=None,  # 0.99,
        detrend=True,
        pair="BTCUSD",
    )
    data, charts, aggregate = main(**inputs)
    filename = f"{inputs['pair']} {inputs['start']} - {inputs['end']} {inputs['window']} overlap={inputs['overlap']} detrend={inputs['detrend']}.enviro"
    file = dict(data=data, charts=charts, aggregate=aggregate, inputs=inputs)
    with open(filename, "wb") as handle:
        pickle.dump(file, handle)
    print(data)
    results = pd.DataFrame(data)
    print("results", results)
    results = results.set_index("figname")
    print("++++++++++++ RESULTS ++++++++++++")
    print(f"{start}, {end}, {window},")
    print("Saved to", filename)
    # print(results)
    sys.exit()
    for chart in charts:
        if np.count_nonzero(chart["frequencies_kept"]) < 3:
            render_charts(**chart)
    render_agg_chart(aggregate["normalized"])
    plt.show()
