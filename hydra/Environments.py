from hydra.utils import timeme
from pandas.core.construction import array
from scipy import stats
import numpy as np
import pandas as pd
from hydra.SimManager import load_prices

# import matplotlib

# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 0)


pair = "XBTUSD"
startDate = pd.to_datetime("2019-06-01")
endDate = pd.to_datetime("2019-07-01")
PROPORTION = 0.025
PRICE_DEVIANCE_CUTOFF = 0.01

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


# def abline(slope, intercept, label=""):
#     """Plot a line from slope and intercept"""
#     axes = plt.gca()
#     x_vals = np.array(axes.get_xlim())
#     y_vals = intercept + slope * x_vals
#     plt.plot(x_vals, y_vals, "--", label=label)


def render_fft_buckets(
    prices,
    valuable_info,
    minute_buckets,
    price_chart=[],
    price_offset=0,
    figname="",
    subplt=None,
):
    # print(prices)
    fft, freqs, index, powers = valuable_info

    # print({key: len(val) for key, val in minute_buckets.items()})
    minute_buckets_avg = {key: np.average(val) for key, val in minute_buckets.items()}

    if subplt is not None:
        subplt.set_title(f"Prices")
        # plt.plot(prices.index, signal.detrend(prices))
        subplt.plot(prices.index, prices, label="price")
        for label, p, *_ in price_chart:
            # coeff = prices.corr(pd.Series(p, index=prices.index))
            # print(label, p, coeff)
            subplt.plot(prices.index, p + price_offset, label=f"{label} ({_})")
    else:
        fig1 = plt.figure(f"{figname} prices")
        # plt.plot(prices.index, signal.detrend(prices))
        plt.plot(prices.index, prices, label="price")
        for label, p, *_ in price_chart:
            # coeff = prices.corr(pd.Series(p, index=prices.index))
            # print(label, p, coeff)
            plt.plot(prices.index, p + price_offset, label=f"{label} ({_})")

    # plt.legend()

    # fig2 = plt.figure(2)
    # plt.plot(freqs, powers)
    # plt.yscale("log")

    # plt.figure(3)
    # plt.plot(minute_buckets_avg.keys(), minute_buckets_avg.values())


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
    # coeff = prices.corr(
    #     pd.Series(
    #         ifft.real,
    #         index=prices.index,
    #     )
    # )

    # coeff_proportion = price_proportion.corr(
    #     pd.Series(
    #         ifft_proportion.real,
    #         index=price_proportion.index,
    #     )
    # )
    # return (key, ifft_proportion, coeff_proportion ** 2, len(fft))
    return (key, ifft_series, len(fft))


def line_gen(slope, intercept, len):
    return np.arange(len) * slope + intercept


@timeme
def main(pair, startDate, endDate, detrend=False):
    time_step = 1 / 60 / 24
    figname = f"{pair} {startDate} - {endDate}"
    prices = load_prices(pair, startDate=startDate, endDate=endDate)["open"]
    price_len = len(prices)
    proportion_len = round(price_len * PROPORTION)
    price_proportion = prices[proportion_len : price_len - proportion_len]

    print(np.asarray(prices))
    if detrend:
        figname = "(D)" + figname
        slope, intercept, *_ = stats.linregress(
            np.array(range(price_len)),
            prices,
        )
        print(slope, intercept, _)
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

    minute_bucket, minute_bucket_df = get_fft_buckets(valuable_info)
    print(minute_bucket_df)

    keys = sorted([k for k in minute_bucket.keys() if k > 0])
    print(f"{len(keys)=}")
    # print("PRICES", prices, prices.shape, key_count)
    df = pd.DataFrame(
        [
            get_ifft_by_key(
                proportion_len, fft, minute_bucket, key, price_proportion_index
            )
            for key in keys
        ],
        columns=["minutes", "inverse detrended price", "num_frequencies"],
    )

    df["correlation"] = df["inverse detrended price"].apply(
        price_detrended_proportion.corr
    )
    df["variance"] = df[["correlation"]] ** 2
    df = df.sort_values(["variance"], ascending=False)
    df["cum variance"] = df[["variance"]].cumsum()

    df["cum detrended price"] = df[["inverse detrended price"]].cumsum()
    df["cum detrended price correlation"] = df["cum detrended price"].apply(
        price_detrended_proportion.corr
    )
    df["cum detrended price variance"] = df["cum detrended price correlation"] ** 2
    print("df", df, df.shape)
    # print("Shits and Giggles", np.sum(df["variance"]), np.sum(df["corr"]))

    # zero = minute_bucket[0]
    # izero = np.fft.ifft(zero)
    # print("ZERO", zero, izero)

    # subset_tuples = list(df.iloc[0:2].itertuples(index=False, name=None))
    cutoffs = []
    for cutoff in np.arange(0.80, 1, 0.0005):
        (
            subset,
            constructed_sum,
            constructed_coeff,
            deviance,
            subset_kept,
            subset_removed,
        ) = variance_calc(cutoff, price_detrended_proportion, trendline_proportion, df)
        cutoffs.append(
            dict(
                cutoff=cutoff,
                subset=subset,
                constructed_sum=constructed_sum,
                constructed_coeff=constructed_coeff,
                deviance=deviance,
                subset_kept=subset_kept,
                subset_removed=subset_removed,
            )
        )

    fig, axs = plt.subplots(2, num=figname)
    fig.suptitle(figname)

    axs[0].set_title("subset deviance")
    subset_removed = [o["subset_removed"] for o in cutoffs]
    deviance = [o["deviance"] for o in cutoffs]
    print(pd.DataFrame(cutoffs))
    axs[0].plot(subset_removed, deviance)
    subset_removed_delta = np.diff(subset_removed)
    deviance_delta = [0, *np.diff(deviance)]
    axs[0].plot(subset_removed, deviance_delta, label="delta")

    def find_closest(arr, val):
        return min(enumerate(arr), key=lambda x: abs(x[1] - val))

    closest_idx, closest_val = find_closest(deviance_delta, 0.01)

    interesting = dict(
        min=min(prices),
        max=max(prices),
        diff=max(prices) / min(prices),
        removed=subset_removed[closest_idx],
    )
    render_fft_buckets(
        price_proportion,
        valuable_info,
        minute_bucket,
        price_chart=[
            ("detrended", price_detrended_proportion),
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
    return interesting


def variance_calc(variance_cutoff, price_detrended, trendline, df):
    # print(f"========== {variance_cutoff} ==========")
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
    constructed_price = constructed_sum + trendline
    price = price_detrended + trendline
    constructed_and_price_diff = np.abs(constructed_price - price) / price
    sorted_diff = np.flip(np.sort(constructed_and_price_diff))
    sorted_diff_len = len(sorted_diff)
    sorted_diff_prop = round(PRICE_DEVIANCE_CUTOFF * sorted_diff_len)
    sorted_diff_subset = sorted_diff[0:sorted_diff_prop]
    deviance = np.mean(sorted_diff_subset)
    return (
        subset,
        constructed_sum,
        constructed_coeff,
        deviance,
        subset_kept,
        subset_removed,
    )
    # render_fft_buckets(prices, valuable_info, minute_bucket, price_chart=[ifft])


if __name__ == "__main__":
    # main(pair, "2018-05-01", "2018-06-01")
    # main(pair, "2018-06-01", "2018-07-01")
    results = pd.DataFrame(
        [
            main(pair, "2020-05-01", "2020-06-01"),
            main(pair, "2020-05-01", "2020-06-01", True),
            # main(pair, "2020-06-01", "2020-07-01"),
            # main(pair, "2020-07-01", "2020-08-01"),
            # main(pair, "2020-08-01", "2020-09-01"),
            # main(pair, "2020-09-01", "2020-10-01"),
            # main(pair, "2020-10-01", "2020-11-01"),
            # main(pair, "2020-11-01", "2020-12-01"),
            # main(pair, "2020-12-01", "2021-01-01"),
        ]
    )
    print("++++++++++++ RESULTS ++++++++++++")
    print(results)
    plt.show()