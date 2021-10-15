# matplotlib.use("TkAgg")
import os
import sys
import traceback
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import ray
from scipy import stats
import seaborn as sns

from hydra.DataLoader import load_prices
from hydra.utils import printd, timeme


# pd.set_option("display.max_rows", 50)
# pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 0)
sns.set_theme()

pair = "XBTUSD"
PROPORTION = 0.025
PRICE_DEVIANCE_CUTOFF = 0.01
PRICE_DEVIANCE_PORTION = 0.01

# 56 @ 99.95
# 165 @ 99.5
# 251 @ 99
# 548 @ 95


@ray.remote
def generate_environment_ray(*args, **kwargs):
    return generate_environment(*args, **kwargs)


@timeme
def generate_environment(
    pair,
    startDate,
    endDate=None,
    detrend=False,
    buckets=False,
    timecap=None,
    normalize_amplitude=True,
):
    startDate = pd.to_datetime(startDate)
    if timecap is not None:
        delta = pd.to_timedelta(timecap)
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
    proportion_len = round(price_len * PROPORTION)
    price_proportion = prices[proportion_len:-proportion_len]

    slope = 0
    intercept = np.mean(prices)
    if detrend:
        figname = "(D)" + figname
        slope, intercept, *_ = stats.linregress(
            np.array(range(price_len)),
            prices,
        )
    trendline = line_gen(slope, intercept, price_len)

    price_detrended = prices - trendline
    price_detrended_proportion = price_detrended[proportion_len:-proportion_len]
    trendline_proportion = trendline[proportion_len:-proportion_len]

    price_index = prices.index
    price_proportion_index = price_proportion.index

    print(f"\n=+=+=+=+=+=+=+=+=+=+= {figname} =+=+=+=+=+=+=+=+=+=+=")
    print(price_detrended)
    valuable_info = transform(price_detrended, time_step)
    fft, freqs, index, powers = valuable_info
    print(f"{len(fft)=} {fft=}")
    print(f"{len(freqs)=} {freqs=}")
    fft_amplitude = (
        np.sqrt((np.abs(fft.real) / len(fft)) ** 2 + (np.abs(fft.imag) / len(fft)) ** 2)
        * 2  # Since the power is split between positive and negative freq, multiply by 2
    )

    if normalize_amplitude:
        fft_amplitude_sum = np.sum(fft_amplitude)
        fft_amplitude = fft_amplitude / fft_amplitude_sum

    print(f"{len(fft_amplitude)=} {fft_amplitude=}")
    fft_amplitude_positive = fft_amplitude[0 : len(fft_amplitude) // 2]
    # printd("PRICES", prices, prices.shape, key_count)
    df: pd.DataFrame = df_from_ifft_variance(
        fft, freqs, price_detrended_proportion, proportion_len
    )

    printd("Sorting variance")
    df = df.sort_values(["variance"], ascending=False)
    printd("Sorted variance")

    (
        majority_price_proportion,  # Price constructed using the most impactful frequencies
        deviances,
        subset_removed,
        frequencies_kept,
    ) = construct_price_significant_frequencies(
        df,
        fft,
        freqs,
        price_detrended_proportion,
        trendline_proportion,
        proportion_len,
    )

    frequencies_kept_amplitude = frequencies_kept * fft_amplitude_positive

    print(
        pd.DataFrame(
            {"Deviance": pd.Series(deviances), "Removed": pd.Series(subset_removed)}
        )
    )
    print(
        pd.DataFrame(
            {
                "Prices": price_detrended_proportion,
                "Majority": majority_price_proportion,
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
    print("kept", frequencies_kept_amplitude)

    interesting = dict(
        index=figname,
        min=min(prices),
        max=max(prices),
        diff=max(prices) / min(prices),
        trend_deviance=deviance_calc(trendline_proportion, 0, price_proportion),
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
        majority_price_proportion=majority_price_proportion + trendline_proportion,
        fft=fft,
        frequencies_kept=frequencies_kept_amplitude,
    )
    return interesting, plotty


@timeme
def construct_price_significant_frequencies(
    df: pd.DataFrame,
    fft,
    freqs,
    price_detrended_proportion: pd.Series,
    trendline_proportion: pd.Series,
    proportion_len: int,
):
    # Running subtraction
    price_detrended_proportion_add = pd.Series(
        np.zeros(len(price_detrended_proportion)),
        index=price_detrended_proportion.index,
    )
    print("price", price_detrended_proportion)
    deviances = [
        deviance_calc(
            price_detrended_proportion_add,
            trendline_proportion,
            price_detrended_proportion,
        )
    ]
    count = 0
    subset_removed = [1]
    freqs_pos = freqs[freqs >= 0]
    frequencies_kept = pd.Series(np.zeros(len(freqs_pos)), index=freqs_pos)
    frequencies_kept.index.rename("frequency", inplace=True)
    print(df)
    base = None
    for idx, frequency, variance in df.itertuples(name=None):
        key = idx + 1
        count += 1
        ifft, base = get_ifft_by_index(fft, key, proportion_len, base)
        price_detrended_proportion_add = price_detrended_proportion_add + ifft
        deviance = deviance_calc(
            price_detrended_proportion_add,
            trendline_proportion,
            price_detrended_proportion,
        )
        deviances.append(deviance)
        subset_removed.append((df.shape[0] - count) / df.shape[0])
        frequencies_kept.loc[frequency] = 1
        if deviance < PRICE_DEVIANCE_CUTOFF:
            return (
                price_detrended_proportion_add,
                deviances,
                subset_removed,
                frequencies_kept,
            )


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


def get_ifft_by_key(fft, minute_bucket, key, price_proportion_index, proportion_len):
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

    ifft_proportion = ifft.real[proportion_len:-proportion_len]
    ifft_series = pd.Series(ifft_proportion, index=price_proportion_index)

    return (key, ifft_series, len(keep_me))


def get_ifft_by_index(fft, idx, proportion_len, base=None):
    if base is None:
        base = np.zeros(len(fft), dtype=np.complex128)
    base[idx] = fft[idx]
    base[-idx] = fft[-idx]
    ifft = np.fft.ifft(base)
    base[idx] = 0
    base[-idx] = 0
    ifft_proportion = ifft[proportion_len:-proportion_len]

    # imaginary_booty = np.argwhere(ifft_proportion.imag > 0.01)
    # if len(imaginary_booty):
    #     printd(f"============ WARNING BOOTY {idx} TOO BIG ============")
    #     printd(ifft_proportion[imaginary_booty])

    return ifft_proportion.real, base


@timeme
def df_from_ifft_variance(
    fft, freqs, price_detrended_proportion: pd.Series, proportion_len
):
    return pd.DataFrame(
        iterate_ifft_variance_np(
            fft, freqs, price_detrended_proportion, proportion_len
        ),
        columns=["frequency", "variance"],
    )


def iterate_ifft_variance_scp(
    fft, freq, price_detrended_proportion: pd.Series, proportion_len
):
    mid = len(fft) // 2
    base = None
    for idx in range(1, mid):
        ifft_proportion, base = get_ifft_by_index(fft, idx, proportion_len, base)
        correlation, pval = stats.pearsonr(
            price_detrended_proportion.values, ifft_proportion
        )
        # print(f"scp {correlation=} {pval=}")
        variance = correlation ** 2
        yield freq[idx], variance


def iterate_ifft_variance_np(
    fft, freq, price_detrended_proportion: pd.Series, proportion_len: pd.Series
):
    mid = len(fft) // 2
    base = None
    for idx in range(1, mid):
        ifft_proportion, base = get_ifft_by_index(fft, idx, proportion_len, base)
        """
        Note: This returns a matrix of the shape
        array([ [1.        , 0.37405599],
                [0.37405599, 1.        ]])
        """
        correlation = np.corrcoef(price_detrended_proportion.values, ifft_proportion)
        correlation = correlation[0, 1]
        # print(f"np {correlation=}")
        variance = correlation ** 2
        yield freq[idx], variance


def iterate_ifft_variance_pd(fft, freq, price_detrended_proportion, proportion_len):
    mid = len(fft) // 2
    base = None
    for idx in range(1, mid):
        ifft_proportion, base = get_ifft_by_index(fft, idx, proportion_len, base)
        ifft_series = pd.Series(ifft_proportion, index=price_detrended_proportion.index)
        correlation = price_detrended_proportion.corr(ifft_series)
        variance = correlation ** 2
        yield freq[idx], variance


def date_line_gen_factory(slope, intercept, originalStartDate, *args, **kwargs):
    def date_line_gen(start, end, interval):
        startDate = pd.to_datetime(start)
        startDelta = startDate - originalStartDate
        startDelta_m = startDelta.total_seconds() / 60

        endDate = pd.to_datetime(end)
        delta = endDate - originalStartDate
        delta_m = delta.total_seconds() / 60

        return line_gen(
            slope, intercept, start=startDelta_m, stop=delta_m, interval=interval
        )

    return date_line_gen


def line_gen(slope, intercept, stop, start=None, interval=1):
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
    majority_price_proportion,
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
            ("majority", majority_price_proportion),
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
    df["min/cycle"] = np.round(1440 / frequencies.index.to_numpy())
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
    for minutes, group in df.groupby("min/cycle"):
        agg.append(group)
        if last is None:
            start = last = minutes
            continue
        if start is None:
            start = minutes
        acc += (minutes - last) / last

        if acc > 0.1:
            label = f"{start}-{minutes}"
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
    bucketed = bucketed.drop(columns=["min/cycle"])
    label_map = bucketed[["label"]]
    label_map = label_map.droplevel(1).drop_duplicates()
    # print(label_map)
    # print(bucketed)
    bucketed = bucketed.groupby(level=0).sum().join(label_map).set_index("label")
    # print(bucketed)
    return bucketed


def process_aggregate(data, chart):
    frequencies_kept_df = pd.DataFrame()
    for d in chart:
        frequencies_kept_df[d["fig_dates"]] = d["frequencies_kept"]
    frequencies_kept_df = frequencies_kept_df.replace(0, np.nan)
    first_idx = frequencies_kept_df.first_valid_index()
    last_idx = frequencies_kept_df.last_valid_index()
    frequencies_kept_df = frequencies_kept_df.loc[first_idx:last_idx]

    frequencies_kept_df = bucket_frequencies(frequencies_kept_df)
    maxes = frequencies_kept_df.max(axis=1)
    frequencies_kept_df = frequencies_kept_df.divide(maxes, axis=0)

    return frequencies_kept_df


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
                    generate_environment_ray.remote(*task["args"], **task["kwargs"])
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
        return [generate_environment(*task["args"], **task["kwargs"]) for task in tasks]


def gen_tasks(start, length, count=8, end=None, overlap=0, detrend=True):
    start = pd.to_datetime(start)
    if end is not None:
        # Todo: see if end is time delta or datetime, use to generate count
        pass
    delta = pd.to_timedelta(length)
    if overlap > 0:
        delta -= pd.to_timedelta(overlap)
    for i in range(count):
        startDate = start + i * delta
        yield {
            "args": (pair, startDate),
            "kwargs": {"detrend": detrend, "timecap": length},
        }


def main(start, timecap, count, overlap, detrend):

    tasks = list(
        gen_tasks(start, timecap, count=count, overlap=overlap, detrend=detrend)
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
    start = "2020-05-01"
    timecap = "1d"
    count = 365
    overlap = "18h"
    detrend = True
    data, charts, aggregate = main(start, timecap, count, overlap, detrend)

    filename = f"{start} {timecap} {count=} {overlap=} {detrend=}.enviro"
    file = dict(data=data, charts=charts, aggregate=aggregate)
    with open(filename, "wb") as handle:
        pickle.dump(file, handle)
    results = pd.DataFrame(data).set_index("index")
    print("++++++++++++ RESULTS ++++++++++++")
    print(results)
    sys.exit()
    for chart in charts:
        if np.count_nonzero(chart["frequencies_kept"]) < 3:
            render_charts(**chart)
    render_agg_chart(aggregate)
    plt.show()
