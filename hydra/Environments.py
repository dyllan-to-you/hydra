from pandas.core.construction import array
from scipy import signal
import numpy as np
import pandas as pd
from hydra.SimManager import load_prices

# import matplotlib

# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
# pd.set_option('display.width', 1000)


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
    prices, valuable_info, minute_buckets, price_chart=[], price_offset=0, figname=""
):
    # print(prices)
    fft, freqs, index, powers = valuable_info

    # print({key: len(val) for key, val in minute_buckets.items()})
    minute_buckets_avg = {key: np.average(val) for key, val in minute_buckets.items()}

    fig1 = plt.figure(f"{figname} prices")
    # plt.plot(prices.index, signal.detrend(prices))
    plt.plot(prices.index, prices, label="price")
    for label, p, coeff, *_ in price_chart:
        # coeff = prices.corr(pd.Series(p, index=prices.index))
        # print(label, p, coeff)
        plt.plot(prices.index, p + price_offset, label=f"{label} ({coeff})")

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


def get_ifft_segments(
    prices, fft, minute_bucket, strip_ranges=[], keep_ranges=[], keep_list=[]
):
    # keys = sorted([k for k in minute_bucket.keys() if k > 0])
    # [zero] = minute_bucket[0]

    # keep_list
    keepers = get_keys(*keep_list, keep=True)
    keep_me = np.concatenate([minute_bucket[key] for key in keepers])

    strip_idx = np.isin(fft, keep_me, assume_unique=True, invert=True)
    keeped = np.copy(fft)
    keeped[strip_idx] = 0
    # print("keep", keep_me)
    # keeped = np.array([x if x in keep_me else 0 for x in fft])
    # print("keeped", ranges, keeped)
    ifft = np.fft.ifft(keeped)
    # zeroed = np.fft.ifft(np.where(keeped == zero, 0, keeped))

    array_len = len(prices)
    array_proportion = round(array_len * PROPORTION)
    price_proportion = prices[array_proportion : array_len - array_proportion]
    ifft_proportion = ifft.copy()[array_proportion : array_len - array_proportion].real
    # zeroed_proportion = zeroed[array_proportion : array_len - array_proportion].real
    # print("prop === ")
    # # print(prices.index, price_proportion.index)
    # print(array_len, array_proportion)
    coeff_proportion = price_proportion.corr(
        pd.Series(
            ifft_proportion,
            index=price_proportion.index,
        )
    )

    # coeff = prices.corr(
    #     pd.Series(
    #         ifft.real,
    #         index=prices.index,
    #     )
    # )
    return (f"kept {keep_list}", ifft_proportion, coeff_proportion ** 2)


def get_ifft_by_key(price_proportion, array_proportion, fft, minute_bucket, key):
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

    ifft_proportion = ifft.real[array_proportion:-array_proportion]
    # coeff = prices.corr(
    #     pd.Series(
    #         ifft.real,
    #         index=prices.index,
    #     )
    # )

    coeff_proportion = price_proportion.corr(
        pd.Series(
            ifft_proportion.real,
            index=price_proportion.index,
        )
    )
    return (key, ifft_proportion, coeff_proportion ** 2, len(fft))


def main(pair, startDate, endDate):
    figname = f"{pair} {startDate} - {endDate}"
    print(f"\n=+=+=+=+=+=+=+=+=+=+= {figname} =+=+=+=+=+=+=+=+=+=+=")
    prices = load_prices(pair, startDate=startDate, endDate=endDate)["open"]
    price_avg = np.mean(prices)
    array_len = len(prices)
    array_proportion = round(array_len * PROPORTION)
    price_proportion = prices[array_proportion : array_len - array_proportion]

    time_step = 1 / 60 / 24

    valuable_info = transform(prices - price_avg, time_step)
    fft, freqs, index, powers = valuable_info

    # print("FFT", fft)

    minute_bucket, minute_bucket_df = get_fft_buckets(valuable_info)
    print(minute_bucket_df)
    # ifft_segments = get_ifft_segments(
    #     prices,
    #     fft,
    #     minute_bucket,  # This is to prevent line wrap
    #     # strip_ranges=[(5, None)],
    #     keep_ranges=[(5, None)],
    # )
    keys = sorted([k for k in minute_bucket.keys() if k > 0])
    print(f"{len(keys)=}")
    # print("PRICES", prices, prices.shape, key_count)
    df = pd.DataFrame(
        [
            get_ifft_by_key(
                price_proportion - price_avg, array_proportion, fft, minute_bucket, key
            )
            for key in keys
        ],
        columns=["minutes", "inversed prices", "variance", "num_frequencies"],
    )
    df = df.sort_values(["variance"], ascending=False)
    df["cum variance"] = df[["variance"]].cumsum()
    # df = df.sort_values(["inversed prices"], ascending=False)

    # df = df.append(
    #     {
    #         "minutes": 999,
    #         "inversed prices": constructed_sum,
    #         "variance": 1,
    #     },
    #     ignore_index=True,
    # )

    # df["corr"] = df["inversed prices"].apply(
    #     lambda x: price_proportion.corr(pd.Series(x, index=price_proportion.index))
    # )
    # df["cum corr"] = df[["corr"]].cumsum()
    # df["cum corr^2"] = df[["cum corr"]] ** 2

    df["cum prices"] = df[["inversed prices"]].cumsum()
    df["cum price variance"] = df["cum prices"].apply(
        lambda x: price_proportion.corr(pd.Series(x, index=price_proportion.index)) ** 2
    )
    print("df", df, df.shape)
    # print("Shits and Giggles", np.sum(df["variance"]), np.sum(df["corr"]))

    # zero = minute_bucket[0]
    # izero = np.fft.ifft(zero)
    # print("ZERO", zero, izero)

    # subset_tuples = list(df.iloc[0:2].itertuples(index=False, name=None))
    cutoffs = []
    for cutoff in np.arange(0.90, 1, 0.0005):
        (
            subset,
            constructed_sum,
            constructed_coeff,
            deviance,
            subset_kept,
            subset_removed,
        ) = variance_calc(cutoff, price_proportion, price_avg, df)
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

    fig0 = plt.figure(f"{figname} subset deviance")
    subset_removed = [o["subset_removed"] for o in cutoffs]
    deviance = [o["deviance"] for o in cutoffs]
    print(pd.DataFrame(cutoffs))
    plt.plot(subset_removed, deviance)
    subset_removed_delta = np.diff(subset_removed)
    deviance_delta = [0, *np.diff(deviance)]
    plt.plot(subset_removed, deviance_delta, label="delta")

    # render_fft_buckets(
    #     price_proportion,
    #     valuable_info,
    #     minute_bucket,
    #     price_chart=[
    #         (
    #             "summed",
    #             cutoffs[-1]["constructed_sum"],
    #             cutoffs[-1]["constructed_coeff"],
    #         ),
    #         *cutoffs[-1]["subset"].itertuples(index=False, name=None),
    #     ],
    #     price_offset=price_avg,
    #     figname=figname,
    # )


def variance_calc(variance_cutoff, price_proportion, price_avg, df):
    # print(f"========== {variance_cutoff} ==========")
    subset = df[df["cum price variance"] <= variance_cutoff]
    subset_kept = len(subset) / len(df)
    subset_removed = 1 - subset_kept
    constructed_sum = np.sum(subset["inversed prices"])
    constructed_coeff = price_proportion.corr(
        pd.Series(
            constructed_sum,
            index=price_proportion.index,
        )
    )
    # print("constructed", constructed_sum, constructed_coeff)
    constructed_price = constructed_sum + price_avg
    constructed_and_price_diff = (
        np.abs(constructed_price - price_proportion) / price_proportion
    )
    sorted_diff = np.flip(np.sort(constructed_and_price_diff))
    sorted_diff_len = len(sorted_diff)
    sorted_diff_prop = round(PRICE_DEVIANCE_CUTOFF * sorted_diff_len)
    sorted_diff_subset = sorted_diff[0:sorted_diff_prop]
    sorted_diff_subset_avg = np.mean(sorted_diff_subset)
    # print(
    #     f"Average of top {PRICE_DEVIANCE_CUTOFF*100}% deviance {sorted_diff_subset_avg * 100}%",
    # )
    # print(
    #     f"Subset Kept { subset_kept * 100 }%",
    # )
    # print(
    #     f"Subset Removed { subset_removed * 100 }%",
    # )
    return (
        subset,
        constructed_sum,
        constructed_coeff,
        sorted_diff_subset_avg,
        subset_kept,
        subset_removed,
    )
    # render_fft_buckets(prices, valuable_info, minute_bucket, price_chart=[ifft])


if __name__ == "__main__":
    main(pair, "2018-05-01", "2018-06-01")
    main(pair, "2018-06-01", "2018-07-01")
    main(pair, "2019-05-01", "2019-06-01")
    main(pair, "2019-06-01", "2019-07-01")
    # plt.show()