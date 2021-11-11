# matplotlib.use("TkAgg")
import os
import sys
import traceback
import pickle
import wave

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import ray
from scipy import stats
import seaborn as sns
from toolz.itertoolz import frequencies

from hydra.DataLoader import load_prices
from hydra.utils import printd, timeme


# pd.set_option("display.max_rows", 50)
# pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 0)
sns.set_theme()

PROPORTION = 0.025
PRICE_DEVIANCE_CUTOFF = 0.01
PRICE_DEVIANCE_PORTION = 0.01

# 56 @ 99.95
# 165 @ 99.5
# 251 @ 99
# 548 @ 95


@ray.remote
def fft_price_analysis_ray(*args, **kwargs):
    return fft_price_analysis(*args, **kwargs)


@timeme
def fft_price_analysis(
    pair,
    startDate,
    endDate=None,
    detrend=False,
    buckets=False,
    window=None,
    normalize_amplitude=True,
):
    startDate = pd.to_datetime(startDate)
    if window is not None:
        delta = pd.to_timedelta(window)
        endDate = startDate + delta
    endDate = pd.to_datetime(endDate)
    time_step = 1 / 60 / 24
    fig_dates = (
        f"{startDate.strftime('%Y-%m-%d %H:%M')} - {endDate.strftime('%Y-%m-%d %H:%M')}"
    )
    figname = f"{pair} {fig_dates}"
    prices = load_prices(pair, startDate=startDate, endDate=endDate)["open"]
    price_len = len(prices)
    # Even numbered price length can result in a 3x speedup!
    if price_len % 2 == 1:
        prices = prices.iloc[:-1]
        price_len = len(prices)
    endDate = prices.index[-1]

    trim_len = round(price_len * PROPORTION)
    price_trim = prices[trim_len:-trim_len]

    slope = 0
    intercept = np.mean(prices)
    if detrend:
        figname = "(D)" + figname
        slope, intercept, *_ = stats.linregress(
            np.array(range(price_len)),
            prices,
        )
    line_gen = date_line_gen_factory(slope, intercept, startDate)
    trendline = line_gen(startDate, endDate)

    price_detrended = prices - trendline
    price_detrended_trim = price_detrended[trim_len:-trim_len]
    trendline_trim = trendline[trim_len:-trim_len]

    price_index = prices.index
    price_trim_index = price_trim.index

    print(f"\n=+=+=+=+=+=+=+=+=+=+= {figname} =+=+=+=+=+=+=+=+=+=+=")
    print("TREND", trendline, len(trendline), startDate, endDate)
    print(price_detrended)
    valuable_info = transform(price_detrended, time_step)
    fft, freqs, index, powers = valuable_info
    print(f"{len(fft)=} {fft=}")
    print(f"{len(freqs)=} {freqs=}")

    # printd("PRICES", prices, prices.shape, key_count)
    df: pd.DataFrame = df_from_ifft_variance(fft, freqs, price_detrended_trim, trim_len)

    df["min/cycle"] = np.round(1440 / df["frequency"].to_numpy())

    (
        approximated_price,  # Price constructed using the most impactful frequencies
        deviances,
        subset_removed,
        frequencies_kept,
    ) = construct_price_significant_frequencies(
        df,
        fft,
        freqs,
        price_detrended,
        trendline,
        trim_len,
    )
    frequencies_kept.name = startDate

    print("FREQUENCY =================================")
    print(frequencies_kept)

    # max_amplitude_ifft=None

    (
        ifft_extrapolated,
        ifft_extrapolated_trended,
        ifft_extrapolated_wavelength,
        ifft_extrapolated_amplitude,
        first_extrapolated,
    ) = extrapolate_ifft(frequencies_kept, fft, prices.index, trendline_gen=line_gen)

    print("RESULTS ==========================")
    print(
        pd.DataFrame(
            {"Deviance": pd.Series(deviances), "Removed": pd.Series(subset_removed)}
        )
    )
    print(
        pd.DataFrame(
            {
                "Prices": price_detrended,
                "Approximated": approximated_price,
            }
        )
    )
    print(
        "df",
        df[
            [
                "frequency",
                "variance",
            ]
        ],
        df.shape,
    )
    print("kept", frequencies_kept)

    interesting = dict(
        index=figname,
        min=min(prices),
        max=max(prices),
        diff=max(prices) / min(prices),
        trend_deviance=deviance_calc(trendline_trim, 0, price_trim),
    )

    plotty = dict(
        startDate=startDate,
        endDate=endDate,
        figname=figname,
        fig_dates=fig_dates,
        subset_removed=subset_removed,
        deviance=deviances,
        price=prices,
        price_detrended=price_detrended,
        trendline=pd.Series(trendline, index=price_index),
        trend_slope=slope,
        trend_intercept=intercept,
        approximated_price=approximated_price + trendline,
        detrended_approximated_price=approximated_price,
        fft=fft,
        frequencies_kept=frequencies_kept,
        extrapolated=ifft_extrapolated_trended,
        extrapolated_wavelength=ifft_extrapolated_wavelength,
        extrapolated_amplitude=ifft_extrapolated_amplitude,
        first_extrapolated=first_extrapolated,
    )
    return interesting, plotty


def extrapolate_ifft(df: pd.DataFrame, fft: np.ndarray, datetime_index, trendline_gen):
    key = df["amplitude"].idxmax()
    row = df.loc[key]
    print("EXTRAPOLATE ========================")
    print(row)

    ifft = pd.Series(get_ifft_by_index(fft, key, 0)[0], index=datetime_index)
    wavelength = pd.to_timedelta(row["min/cycle"], unit="m")
    last_cycle_start = datetime_index[-1] - wavelength
    last_cycle = ifft[last_cycle_start:]
    last_max_idx = last_cycle.idxmax()
    last_min_idx = last_cycle.idxmin()
    delta = abs(last_max_idx - last_min_idx)
    if delta != wavelength / 2:
        print("WARNING: delta != wavelength/2", delta, wavelength)

    endpoint = datetime_index[-1] + (datetime_index[-1] - datetime_index[0])

    print(f"{last_min_idx=} {last_max_idx=} {endpoint=}")
    trendline = trendline_gen(datetime_index[0], datetime_index[-1], index=True)

    extrapolated_trendline = trendline_gen(
        min(last_min_idx, last_max_idx) + wavelength,
        endpoint,
        interval=delta.total_seconds() / 60,
        index=True,
    )

    extras = []
    start_pole = 1 if last_max_idx < last_min_idx else -1
    for date in extrapolated_trendline.index:
        extras.append(row["amplitude"] * start_pole)
        start_pole *= -1

    extrapolated = pd.Series(extras, extrapolated_trendline.index)
    extrapolated_trended = extrapolated + extrapolated_trendline
    print("LAST CYCLE", trendline.loc[[last_max_idx, last_min_idx]])
    print("EXTRPOLATD", extrapolated)
    return (
        extrapolated,
        extrapolated_trended,
        wavelength,
        row["amplitude"],
        (extrapolated_trended[0], extrapolated_trended.index[0], extrapolated[0] > 0),
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
    trim_len: int,
):
    price_detrended_add = pd.Series(
        np.zeros(len(price_detrended)),
        index=price_detrended.index,
    )
    print("price", price_detrended)
    deviances = [
        deviance_calc(
            price_detrended_add[trim_len:-trim_len],
            trendline[trim_len:-trim_len],
            price_detrended[trim_len:-trim_len],
        )
    ]
    count = 0
    subset_removed = [1]

    df["keep"] = np.zeros(len(df), dtype=bool)
    print(df)

    empty_fft = None
    for key, frequency, *_ in df.sort_values(["variance"], ascending=False).itertuples(
        name=None
    ):
        count += 1
        ifft, empty_fft = get_ifft_by_index(fft, key, 0, empty_fft)
        price_detrended_add = price_detrended_add + ifft
        deviance = deviance_calc(
            price_detrended_add[trim_len:-trim_len],
            trendline[trim_len:-trim_len],
            price_detrended[trim_len:-trim_len],
        )
        deviances.append(deviance)
        subset_removed.append((df.shape[0] - count) / df.shape[0])
        df.loc[key, "keep"] = True
        if deviance < PRICE_DEVIANCE_CUTOFF:
            keep = df[df["keep"]].drop("keep", axis=1)
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
    print("fft_res", fft_res)
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


def get_ifft_by_key(fft, minute_bucket, key, price_trim_index, trim_len):
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

    ifft_trim = ifft.real[trim_len:-trim_len]
    ifft_series = pd.Series(ifft_trim, index=price_trim_index)

    return (key, ifft_series, len(keep_me))


def get_ifft_by_index(fft, idx, trim_len, empty_fft=None):
    if empty_fft is None:
        empty_fft = np.zeros(len(fft), dtype=np.complex128)
    empty_fft[idx] = fft[idx]
    empty_fft[-idx] = fft[-idx]
    ifft = np.fft.ifft(empty_fft)
    empty_fft[idx] = 0
    empty_fft[-idx] = 0
    if trim_len == 0:
        ifft_trim = ifft
    else:
        ifft_trim = ifft[trim_len:-trim_len]

    imaginary_booty = np.argwhere(ifft_trim.imag > 0.01)
    if len(imaginary_booty):
        printd(f"============ WARNING BOOTY {idx} TOO BIG ============")
        printd(ifft_trim[imaginary_booty])

    return ifft_trim.real, empty_fft


@timeme
def df_from_ifft_variance(fft, freqs, price_detrended_trim: pd.Series, trim_len):
    return pd.DataFrame(
        iterate_ifft_variance_np(fft, freqs, price_detrended_trim, trim_len),
        columns=["key", "frequency", "variance", "amplitude", "amplitude_normalized"],
    ).set_index("key")


def iterate_ifft_variance_np(
    fft, freq, price_detrended_trim: pd.Series, trim_len: pd.Series
):
    mid = len(fft) // 2
    empty_fft = None
    fft_amplitude = (
        np.sqrt((np.abs(fft.real) / len(fft)) ** 2 + (np.abs(fft.imag) / len(fft)) ** 2)
        * 2  # Since the power is split between positive and negative freq, multiply by 2
    )
    fft_amplitude_sum = np.sum(fft_amplitude)
    fft_amplitude_normalized = fft_amplitude / fft_amplitude_sum
    for idx in range(1, mid):
        ifft_trim, empty_fft = get_ifft_by_index(fft, idx, trim_len, empty_fft)
        """
        Note: This returns a matrix of the shape
        array([ [1.        , 0.37405599],
                [0.37405599, 1.        ]])
        """
        correlation = np.corrcoef(price_detrended_trim.values, ifft_trim)
        correlation = correlation[0, 1]
        # print(f"np {correlation=}")
        variance = correlation ** 2
        yield idx, freq[idx], variance, fft_amplitude[idx], fft_amplitude_normalized[
            idx
        ]


def date_line_gen_factory(slope, intercept, originalStartDate, *args, **kwargs):
    def date_line_gen(start, end, interval=1, index=False, inclusive=True):
        interval = int(interval)
        startDate = pd.to_datetime(start)
        startDelta = startDate - originalStartDate
        startDelta_m = startDelta.total_seconds() / 60

        endDate = pd.to_datetime(end)
        delta = endDate - originalStartDate
        delta_m = delta.total_seconds() / 60

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
        idx = pd.date_range(startDate, endDate, freq=f"{interval}min")
        return pd.Series(line, idx)

    return date_line_gen


def line_generator(
    slope, intercept, stop, start=None, interval=1, inclusive=True
) -> np.ndarray:
    interval = int(interval)
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
    sorted_diff_prop = round(PRICE_DEVIANCE_PORTION * sorted_diff_len)
    sorted_diff_subset = sorted_diff[0:sorted_diff_prop]
    deviance = np.mean(sorted_diff_subset)
    return deviance


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
    # df["min/cycle"] = frequencies.index  # np.round(1440 / frequencies.index.to_numpy())
    df = df.iloc[::-1]
    # df["diff"] = df["min/cycle"].diff()
    # df["diff%"] = df["diff"].div(df["min/cycle"].shift(1))
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
    # bucketed = bucketed.drop(columns=["min/cycle"])
    label_map = bucketed[["label"]]
    label_map = label_map.droplevel(1).drop_duplicates()
    # print(label_map)
    # print(bucketed)
    bucketed = bucketed.groupby(level=0).sum().join(label_map).set_index("label")
    # print(bucketed)
    return bucketed


def process_aggregate(data, chart):
    print("PROCESS AGGREGATE =======")

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
    print(frequencies_kept_df.index)
    # width = frequencies_kept_df.index[1] - frequencies_kept_df.index[0]
    # axs.bar(frequencies_kept_df.index, frequencies_kept_df["sum"], width=width)
    current_cmap = matplotlib.cm.get_cmap("afmhot_r").copy()
    current_cmap.set_bad(color="black")
    # afmhot_r
    sns.heatmap(frequencies_kept_df, cmap=current_cmap, ax=axs, robust=True)
    axs.tick_params(axis="x", labelrotation=-90)


RATE_LIMIT = 32


@timeme
def parallel_handler(tasks):
    if sys.argv[-1] == "ray":
        try:
            ray.init(include_dashboard=False, local_mode=False)
            # cProfile.run("main()")

            result_refs = []
            for idx, task in enumerate(tasks):
                if len(result_refs) > RATE_LIMIT:
                    ray.wait(result_refs, num_returns=idx - RATE_LIMIT)
                result_refs.append(
                    fft_price_analysis_ray.remote(*task["args"], **task["kwargs"])
                )
            return ray.get(result_refs)
        except KeyboardInterrupt:
            printd("Interrupted")
            return None
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            printd(f"{e}")
            return None
        finally:
            ray.shutdown()
    else:
        return [fft_price_analysis(*task["args"], **task["kwargs"]) for task in tasks]


def gen_tasks(start, end, window, overlap=None, detrend=True, pair="XBTUSD"):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    if end is not None:
        # Todo: see if end is time delta or datetime, use to generate count
        pass
    delta = pd.to_timedelta(window)
    if overlap is not None:
        delta *= 1 - overlap
        delta = delta.round("1min")
    startDate = start
    i = 0
    while startDate < end:
        startDate = start + i * delta
        i += 1
        i = i + 1
        yield {
            "args": (pair, startDate),
            "kwargs": {"detrend": detrend, "window": window},
        }


def main(start, end, window, overlap, detrend, pair):

    tasks = list(
        gen_tasks(
            start, end, window=window, overlap=overlap, detrend=detrend, pair=pair
        )
    )
    results = parallel_handler(tasks)
    if results is None:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    data, charts = zip(*results)

    aggregates = process_aggregate(data, charts)
    return data, charts, aggregates


if __name__ == "__main__":
    inputs = dict(
        start="2019-01-01",
        end="2020-12-01",
        window="1d",
        overlap=0.99,
        detrend=True,
        pair="XBTUSD",
    )
    data, charts, aggregate = main(**inputs)

    filename = f"{inputs['start']} - {inputs['end']} {inputs['window']} overlap={inputs['overlap']} detrend={inputs['detrend']}.enviro"
    file = dict(data=data, charts=charts, aggregate=aggregate, inputs=inputs)
    with open(filename, "wb") as handle:
        pickle.dump(file, handle)
    results = pd.DataFrame(data).set_index("index")
    print("++++++++++++ RESULTS ++++++++++++")
    # print(results)
    sys.exit()
    for chart in charts:
        if np.count_nonzero(chart["frequencies_kept"]) < 3:
            render_charts(**chart)
    render_agg_chart(aggregate["normalized"])
    plt.show()
