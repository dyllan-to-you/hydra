# matplotlib.use("TkAgg")
import os
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
from scipy import stats

from hydra.DataLoader import load_prices
from hydra.mpl_charts import heatmap
from hydra.utils import printd, timeme

# import matplotlib


pd.set_option("display.max_rows", 50)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 0)


pair = "XBTUSD"
PROPORTION = 0.025
PRICE_DEVIANCE_CUTOFF = 0.01
PRICE_DEVIANCE_PORTION = 0.01

# 56 @ 99.95
# 165 @ 99.5
# 251 @ 99
# 548 @ 95


@ray.remote
def main_ray(*args, **kwargs):
    return main(*args, **kwargs)


@timeme
def main(pair, startDate, endDate, detrend=False, buckets=False, timecap=None):
    if timecap is not None:
        delta = pd.to_timedelta(timecap)
        endDate = pd.to_datetime(startDate) + delta
    time_step = 1 / 60 / 24
    figname = f"{pair} {startDate} - {endDate}"
    prices = load_prices(pair, startDate=startDate, endDate=endDate)["open"]
    price_len = len(prices)
    # Even numbered price length can result in a 3x speedup!
    if price_len % 2 == 1:
        prices = prices.iloc[:-1]
        price_len = len(prices)
    proportion_len = round(price_len * PROPORTION)
    get_proportion = proportion_factory(proportion_len)
    price_proportion = get_proportion(prices)

    if detrend:
        figname = "(D)" + figname
        slope, intercept, *_ = stats.linregress(
            np.array(range(price_len)),
            prices,
        )
        trendline = line_gen(slope, intercept, price_len)
    else:
        trendline = np.full(price_len, np.mean(prices))

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
    printd("Calculating Amplitude")
    fft_amplitude = (
        np.sqrt((np.abs(fft.real) / len(fft)) ** 2 + (np.abs(fft.imag) / len(fft)) ** 2)
        * 2
    )  # Since the power is split between positive and negative freq, multiply by 2
    printd("Calculated Amplitude")
    # printd("PRICES", prices, prices.shape, key_count)
    df: pd.DataFrame = None
    if buckets:
        minute_bucket, minute_bucket_df = get_fft_buckets(valuable_info)
        print(minute_bucket_df)

        keys = sorted([k for k in minute_bucket.keys() if k > 0])
        printd(f"{len(keys)=}")
        df = pd.DataFrame(
            [
                get_ifft_by_key(
                    proportion_len, fft, minute_bucket, key, price_proportion_index
                )
                for key in keys
            ],
            columns=["frequency", "inverse detrended price", "num_frequencies"],
        )
    else:
        df = df_from_ifft_variance(
            proportion_len, price_detrended_proportion, fft, freqs
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
        proportion_len,
        price_proportion,
        price_detrended_proportion,
        trendline_proportion,
        fft,
        freqs,
        df,
    )
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

    interesting = dict(
        index=figname,
        min=min(prices),
        max=max(prices),
        diff=max(prices) / min(prices),
        trend_deviance=deviance_calc(trendline_proportion, 0, price_proportion),
    )

    plotty = dict(
        figname=figname,
        subset_removed=subset_removed,
        deviance=deviances,
        price_proportion=price_proportion,
        price_detrended_proportion=price_detrended_proportion,
        trendline_proportion=trendline_proportion,
        majority_price_proportion=majority_price_proportion + trendline_proportion,
        fft=fft,
        frequencies_kept=frequencies_kept,
    )
    return interesting, plotty


@timeme
def construct_price_significant_frequencies(
    proportion_len,
    price_proportion,
    price_detrended_proportion,
    trendline_proportion,
    fft,
    freqs,
    df,
):
    # Running subtraction
    price_detrended_proportion_add = pd.Series(
        np.zeros(len(price_proportion)), index=price_proportion.index
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
    print(df)
    for idx, frequency, variance in df.itertuples(name=None):
        key = idx + 1
        count += 1
        ifft = get_ifft_by_index(fft, key, proportion_len, price_proportion.index)
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
    # printd("NUM SAMPLES,", values.shape, values)
    freqs = np.fft.fftfreq(values.size, time_step)
    index = np.argsort(freqs)
    fft, powers = get_fft(values, time_step)
    # printd(freqs)
    return fft, freqs, index, powers


def get_fft(values, time_step):
    fft = np.fft.fft(values)
    fft[0] = 0
    powers = fft.real ** 2
    # eval_fft(fft, time_step)
    # printd(fft, powers)
    return fft, powers


def eval_fft(fft, time_step):
    printd("TRANFORM")
    printd(
        "amp",
        np.sqrt(
            (np.abs(fft.real) / len(fft)) ** 2 + (np.abs(fft.imag) / len(fft)) ** 2,
        )
        * 2,
    )


def chart_fft(ffta, fftb, time_step):
    eval_fft(ffta, time_step)
    eval_fft(fftb, time_step)


@timeme
def get_fft_buckets(valuable_info):
    _fft, freqs, index, powers = valuable_info
    # printd("VALUABLE", valuable_info)

    freqs = freqs[index]
    fft = _fft[index]
    # ([zero],) = np.where(freqs == 0)
    # freqs = freqs[zero + 1 :]
    # fft = fft[zero + 1 :]
    minute_buckets = {}
    fft_res = pd.DataFrame(dict(power=fft, freqs=freqs))
    # printd("fft,freqs", fft_res)
    for idx, freq in enumerate(freqs):
        power = fft[idx]
        min_per_cycle = None
        if freq == 0:
            min_per_cycle = 0
        else:
            min_per_cycle = round(1440 / freq)  # if freq != 0 else 0
        # printd(idx, freq, power, min_per_cycle)
        bucket = minute_buckets.setdefault(freq, [])
        bucket.append(power)
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


def get_ifft_by_key(proportion_len, fft, minute_bucket, key, price_proportion_index):
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


def get_ifft_by_index(fft, idx, proportion_len, price_proportion_index):
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

    ifft_series = pd.Series(
        ifft_proportion.real,
        index=price_proportion_index,
    )
    return ifft_series


@timeme
def df_from_ifft_variance(
    proportion_len, price_detrended_proportion: pd.Series, fft, freqs
):
    return pd.DataFrame(
        iterate_ifft_variance_np(
            fft, freqs, proportion_len, price_detrended_proportion
        ),
        columns=["frequency", "variance"],
    )


def iterate_ifft_variance_scp(
    fft, freq, proportion_len, price_detrended_proportion: pd.Series
):
    mid = len(fft) // 2
    base = np.zeros(len(fft), dtype=np.complex128)
    for idx in range(1, mid):
        base[idx] = fft[idx]
        base[-idx] = fft[-idx]
        ifft = np.fft.ifft(base)
        base[idx] = 0
        base[-idx] = 0
        ifft_proportion = ifft.real[proportion_len:-proportion_len]
        correlation, pval = stats.pearsonr(
            price_detrended_proportion.values, ifft_proportion
        )
        # print(f"scp {correlation=} {pval=}")
        variance = correlation ** 2
        yield freq[idx], variance


def iterate_ifft_variance_np(
    fft, freq, proportion_len, price_detrended_proportion: pd.Series
):
    mid = len(fft) // 2
    base = np.zeros(len(fft), dtype=np.complex128)
    for idx in range(1, mid):
        base[idx] = fft[idx]
        base[-idx] = fft[-idx]
        ifft = np.fft.ifft(base)
        base[idx] = 0
        base[-idx] = 0
        ifft_proportion = ifft.real[proportion_len:-proportion_len]
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


def iterate_ifft_variance_pd(fft, freq, proportion_len, price_detrended_proportion):
    mid = len(fft) // 2
    base = np.zeros(len(fft), dtype=np.complex128)
    for idx in range(1, mid):
        base[idx] = fft[idx]
        base[-idx] = fft[-idx]
        ifft = np.fft.ifft(base)
        base[idx] = 0
        base[-idx] = 0
        ifft_proportion = ifft.real[proportion_len:-proportion_len]
        ifft_series = pd.Series(ifft_proportion, index=price_detrended_proportion.index)
        correlation = price_detrended_proportion.corr(ifft_series)
        variance = correlation ** 2
        yield freq[idx], variance


def line_gen(slope, intercept, len):
    return np.arange(len) * slope + intercept


def proportion_factory(proportion_len):
    return lambda x: x[proportion_len:-proportion_len]


def additive_variance_calc(variance_cutoff, price_detrended, trendline, df):
    # printd(f"========== {variance_cutoff} ==========")
    df["cum variance"] = df[["variance"]].cumsum()

    df["cum detrended price"] = df[["inverse detrended price"]].cumsum()
    df["cum detrended price correlation"] = df["cum detrended price"].apply(
        price_detrended.corr
    )
    df["cum detrended price variance"] = df["cum detrended price correlation"] ** 2
    subset = df[df["cum detrended price variance"] <= variance_cutoff]
    subset_kept = len(subset) / len(df)
    subset_removed = 1 - subset_kept
    constructed_sum = np.sum(subset["inverse detrended price"])
    constructed_coeff = price_detrended.corr(
        pd.Series(
            constructed_sum,
            index=price_detrended.index,
        )
    )
    # printd("constructed", constructed_sum, constructed_coeff)
    deviance = deviance_calc(constructed_sum, trendline, price_detrended)
    return (
        subset,
        constructed_sum,
        constructed_coeff,
        deviance,
        subset_kept,
        subset_removed,
    )


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
    subset_removed,
    deviance,
    price_proportion,
    price_detrended_proportion,
    trendline_proportion,
    majority_price_proportion,
    fft,
    frequencies_kept,
    **kwargs,
):
    fig, axs = plt.subplots(2, num=figname)
    fig.suptitle(figname)
    axs[0].set_title("subset deviance")
    axs[0].plot(subset_removed, deviance, ".")
    render_price_chart(
        price_proportion,
        price_chart=[
            # ("detrended", price_detrended_proportion),
            ("majority", majority_price_proportion),
            ("trendline", trendline_proportion),
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
        figname=figname,
        ax=axs[1],
    )

    # axs[2].set_title("Important Frequencies")
    # printd(f"{figname}: kept", frequencies_kept)
    # width = frequencies_kept.index[1] - frequencies_kept.index[0]
    # axs[2].bar(frequencies_kept.index, frequencies_kept.values, width=width)


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


def render_agg_chart(data):
    fig, axs = plt.subplots(1, num="Aggregate")
    fig.suptitle("Aggregate")
    frequencies_kept_df = pd.DataFrame()
    for d in data:
        frequencies_kept_df[d["figname"]] = d["frequencies_kept"]
    print("agg: kept", frequencies_kept_df)
    print(
        frequencies_kept_df.to_numpy(),
        frequencies_kept_df.index,
        frequencies_kept_df.columns,
    )
    heatmap(
        frequencies_kept_df.to_numpy(),
        frequencies_kept_df.index,
        frequencies_kept_df.columns,
        ax=axs,
    )
    # axs.bar(frequencies_kept_agg.index, frequencies_kept_agg.values, width=width)


RATE_LIMIT = 32


@timeme
def supermain(tasks):
    if sys.argv[-1] == "ray":
        try:
            ray.init(include_dashboard=False, local_mode=False)
            # cProfile.run("main()")

            result_refs = []
            for idx, task in enumerate(tasks):
                if len(result_refs) > RATE_LIMIT:
                    ray.wait(result_refs, num_returns=idx - RATE_LIMIT)
                result_refs.append(main_ray.remote(*task["args"], **task["kwargs"]))
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
        return [main(*task["args"], **task["kwargs"]) for task in tasks]


if __name__ == "__main__":
    results = None
    tasks = [
        {
            "args": (pair, "2020-05-01", "2020-06-01"),
            "kwargs": {"detrend": True, "timecap": "30d"},
        },
        # {
        #     "args": (pair, "2020-06-01", "2020-07-01"),
        #     "kwargs": {"detrend": True, "timecap": "30d"},
        # },
        # {
        #     "args": (
        #         pair,
        #         "2020-07-01",
        #         "2020-08-01",
        #     ),
        #     "kwargs": {"detrend": True, "timecap": "30d"},
        # },
        # {
        #     "args": (
        #         pair,
        #         "2020-08-01",
        #         "2020-09-01",
        #     ),
        #     "kwargs": {"detrend": True, "timecap": "30d"},
        # },
        # {
        #     "args": (
        #         pair,
        #         "2020-09-01",
        #         "2020-10-01",
        #     ),
        #     "kwargs": {"detrend": True, "timecap": "30d"},
        # },
        # {
        #     "args": (
        #         pair,
        #         "2020-10-01",
        #         "2020-11-01",
        #     ),
        #     "kwargs": {"detrend": True, "timecap": "30d"},
        # },
        # {
        #     "args": (
        #         pair,
        #         "2020-11-01",
        #         "2020-12-01",
        #     ),
        #     "kwargs": {"detrend": True, "timecap": "30d"},
        # },
        # {
        #     "args": (
        #         pair,
        #         "2020-12-01",
        #         "2021-01-01",
        #     ),
        #     "kwargs": {"detrend": True, "timecap": "30d"},
        # },
    ]
    results = supermain(tasks)
    if results is None:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    data, charts = zip(*results)
    results = pd.DataFrame(data).set_index("index")
    print("++++++++++++ RESULTS ++++++++++++")
    print(results)
    # for chart in charts:
    #     render_charts(**chart)
    # render_agg_chart(charts)
    # plt.show()
