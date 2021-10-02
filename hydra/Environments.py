# matplotlib.use("TkAgg")
from math import isclose
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
from pandas.core.construction import array
from scipy import stats

from hydra.SimManager import load_prices
from hydra.utils import timeme

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


def transform(prices, time_step, detrend=False):
    values = prices if detrend is False else signal.detrend(prices)
    # print("NUM SAMPLES,", values.shape, values)
    fft = np.fft.fft(values)
    freqs = np.fft.fftfreq(values.size, time_step)
    index = np.argsort(freqs)
    fft[0] = 0
    powers = np.abs(fft) ** 2
    return fft, freqs, index, powers


def get_fft_buckets(valuable_info):
    _fft, freqs, index, powers = valuable_info
    # print("VALUABLE", valuable_info)

    freqs = freqs[index]
    fft = _fft[index]
    # ([zero],) = np.where(freqs == 0)
    # freqs = freqs[zero + 1 :]
    # fft = fft[zero + 1 :]
    minute_buckets = {}
    fft_res = pd.DataFrame(dict(power=fft, freqs=freqs))
    # print("fft,freqs", fft_res)
    for idx, freq in enumerate(freqs):
        power = fft[idx]
        min_per_cycle = None
        if freq == 0:
            min_per_cycle = 0
        else:
            min_per_cycle = round(1440 / freq)  # if freq != 0 else 0
        # print(idx, freq, power, min_per_cycle)
        bucket = minute_buckets.setdefault(freq, [])
        bucket.append(power)
    # print("Keys", sorted([k for k in minute_buckets.keys() if k > 0]))
    minute_bucket_df = pd.DataFrame(
        dict(minutes=minute_buckets.keys(), fft=minute_buckets.values())
    )
    minute_bucket_df["powers"] = [
        np.abs(power) ** 2 for power in minute_bucket_df["fft"]
    ]
    minute_bucket_df["count"] = [len(power) for power in minute_bucket_df["fft"]]
    minute_bucket_df["cum_count"] = minute_bucket_df[["count"]].cumsum()
    # print(minute_bucket_df)
    return minute_buckets, minute_bucket_df


def get_keys(*arr, keep=False):
    _arr = list(arr)
    # if keep:
    #     _arr += [0]
    return _arr + [x * -1 for x in _arr]


def get_ifft_by_key(proportion_len, fft, minute_bucket, key, price_proportion_index):
    # print("GET IFFT BY KEY", key)
    # [zero] = minute_bucket[0]
    # print("ZERO", zero)
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
    ifft_proportion = ifft.real[proportion_len:-proportion_len]
    ifft_series = pd.Series(ifft_proportion, index=price_proportion_index)
    return ifft_series


def iterate_ifft_variance(fft, freq, proportion_len, price_detrended_proportion):
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


@ray.remote
def main_ray(pair, startDate, endDate, detrend=False):
    return main(pair, startDate, endDate, detrend)


@timeme
def main(pair, startDate, endDate, detrend=False, buckets=False):
    time_step = 1 / 60 / 24
    figname = f"{pair} {startDate} - {endDate}"
    prices = load_prices(pair, startDate=startDate, endDate=endDate)["open"]
    price_len = len(prices)
    proportion_len = round(price_len * PROPORTION)
    price_proportion = prices[proportion_len : price_len - proportion_len]

    if detrend:
        figname += "(D)"
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

    valuable_info = transform(price_detrended, time_step)
    fft, freqs, index, powers = valuable_info

    # print("FFT", fft)

    # print("PRICES", prices, prices.shape, key_count)
    df = None
    if buckets:
        minute_bucket, minute_bucket_df = get_fft_buckets(valuable_info)
        print(minute_bucket_df)

        keys = sorted([k for k in minute_bucket.keys() if k > 0])
        print(f"{len(keys)=}")
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
        df = pd.DataFrame(
            iterate_ifft_variance(
                fft, freqs, proportion_len, price_detrended_proportion
            ),
            columns=["frequency", "variance"],
        )

    df = df.sort_values(["variance"], ascending=False)

    """
    X Get variance when doing ifft
    X throw out the inverse detrended price
    X sort using the variance from least -> most impactful
    - regenerate ifft of least impactful, remove from price, see resulting accuracy/deviance, repeat until accuracy falls too far
    """
    # Running subtraction
    price_detrended_proportion_add = pd.Series(
        np.zeros(len(price_proportion)), index=price_proportion_index
    )
    # Cutoff
    price_detrended_proportion_add_majority = pd.Series(
        np.zeros(len(price_proportion)), index=price_proportion_index
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
    for idx, frequency, variance in df.itertuples(name=None):
        key = idx + 1
        count += 1
        ifft = get_ifft_by_index(fft, key, proportion_len, price_proportion_index)
        price_detrended_proportion_add += ifft
        deviance = deviance_calc(
            price_detrended_proportion_add,
            trendline_proportion,
            price_detrended_proportion,
        )
        deviances.append(deviance)
        subset_removed.append((df.shape[0] - count) / df.shape[0])
        if deviance >= PRICE_DEVIANCE_CUTOFF:
            price_detrended_proportion_add_majority += ifft
        else:
            print(
                deviance,
                subset_removed[-1],
                count,
                df.shape[0],
            )
            break
    print(
        pd.DataFrame(
            {"Deviance": pd.Series(deviances), "Removed": pd.Series(subset_removed)}
        )
    )
    print(
        pd.DataFrame(
            {
                "Prices": price_detrended_proportion,
                "Majority": price_detrended_proportion_add_majority,
            }
        )
    )
    print(
        "df",
        df[
            [
                "frequency",
                # "correlation",
                "variance",
                # "cum variance",
                # "cum detrended price correlation",
                # "cum detrended price variance",
            ]
        ],
        df.shape,
    )
    # return
    # cutoffs = []
    # for cutoff in np.arange(0.80, 1, 0.0005):
    #     (
    #         subset,
    #         constructed_sum,
    #         constructed_coeff,
    #         deviance,
    #         subset_kept,
    #         subset_removed,
    #     ) = additive_variance_calc(
    #         cutoff, price_detrended_proportion, trendline_proportion, df
    #     )
    #     cutoffs.append(
    #         dict(
    #             cutoff=cutoff,
    #             subset=subset,
    #             constructed_sum=constructed_sum,
    #             constructed_coeff=constructed_coeff,
    #             deviance=deviance,
    #             subset_kept=subset_kept,
    #             subset_removed=subset_removed,
    #         )
    #     )

    # print(len(cutoffs), "")
    # subset_removed = [o["subset_removed"] for o in cutoffs]
    # deviance = [o["deviance"] for o in cutoffs]
    # print(
    #     pd.DataFrame(cutoffs)[
    #         ["cutoff", "constructed_coeff", "deviance", "subset_kept", "subset_removed"]
    #     ]
    # )

    # def find_closest_under(arr, val):
    #     return min(
    #         filter(lambda x: x[1] < val, enumerate(arr)), key=lambda x: abs(x[1] - val)
    #     )

    # closest_idx, closest_val = find_closest_under(deviance, 0.01)

    interesting = dict(
        index=figname,
        min=min(prices),
        max=max(prices),
        diff=max(prices) / min(prices),
        trend_deviance=deviance_calc(trendline_proportion, 0, price_proportion),
    )

    plotty = (
        figname,
        subset_removed,
        deviances,
        price_proportion,
        price_detrended_proportion,
        trendline_proportion,
        price_detrended_proportion_add_majority + trendline_proportion,
    )
    return interesting, plotty


def additive_variance_calc(variance_cutoff, price_detrended, trendline, df):
    # print(f"========== {variance_cutoff} ==========")
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
    # print("constructed", constructed_sum, constructed_coeff)
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


def render_charts(
    figname,
    subset_removed,
    deviance,
    price_proportion,
    price_detrended_proportion,
    trendline_proportion,
    majority_price_proportion,
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
        subplt=axs[1],
    )


def render_price_chart(
    prices,
    price_chart=[],
    price_offset=0,
    figname="",
    subplt=None,
):
    if subplt is not None:
        subplt.set_title(f"Prices")
        # plt.plot(prices.index, signal.detrend(prices))
        subplt.plot(prices.index, prices, label="price")
        for label, p, *_ in price_chart:
            # coeff = prices.corr(pd.Series(p, index=prices.index))
            # print(label, p, coeff)
            subplt.plot(prices.index, p + price_offset, label=f"{label} ({_})")
    else:
        fig = plt.figure(f"{figname} prices")
        plt.plot(prices.index, prices, label="price")
        for label, p, *_ in price_chart:
            plt.plot(prices.index, p + price_offset, label=f"{label} ({_})")


RATE_LIMIT = 32


@timeme
def supermain(tasks):
    if sys.argv[-1] == "ray":
        ray.init(include_dashboard=False, local_mode=False)

        result_refs = []
        for idx, task in enumerate(tasks):
            if len(result_refs) > RATE_LIMIT:
                ray.wait(result_refs, num_returns=idx - RATE_LIMIT)
            result_refs.append(main_ray.remote(*task["args"], **task["kwargs"]))
        results = ray.get(result_refs)
    else:
        results = [main(*task["args"], **task["kwargs"]) for task in tasks]
    return results


if __name__ == "__main__":
    results = None
    tasks = [
        {
            "args": (pair, "2020-05-01", "2020-06-01"),
            "kwargs": {"detrend": True},
        },
        {
            "args": (pair, "2020-06-01", "2020-07-01"),
            "kwargs": {"detrend": True},
        },
        {
            "args": (
                pair,
                "2020-07-01",
                "2020-08-01",
            ),
            "kwargs": {"detrend": True},
        },
        {
            "args": (
                pair,
                "2020-08-01",
                "2020-09-01",
            ),
            "kwargs": {"detrend": True},
        },
        {
            "args": (
                pair,
                "2020-09-01",
                "2020-10-01",
            ),
            "kwargs": {"detrend": True},
        },
        {
            "args": (
                pair,
                "2020-10-01",
                "2020-11-01",
            ),
            "kwargs": {"detrend": True},
        },
        {
            "args": (
                pair,
                "2020-11-01",
                "2020-12-01",
            ),
            "kwargs": {"detrend": True},
        },
        {
            "args": (
                pair,
                "2020-12-01",
                "2021-01-01",
            ),
            "kwargs": {"detrend": True},
        },
    ]
    results = supermain(tasks)

    data, charts = zip(*results)
    results = pd.DataFrame(data).set_index("index")
    print("++++++++++++ RESULTS ++++++++++++")
    print(results)
    for chart in charts:
        render_charts(*chart)
    plt.show()
