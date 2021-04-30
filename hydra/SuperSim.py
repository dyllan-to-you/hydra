import pathlib
import argparse
import time
import numpy as np
import pathlib
from numba import njit
import vectorbt as vbt
from vectorbt.portfolio.nb import create_order_nb
from vectorbt.signals.factory import SignalFactory
from hydra.utils import printd
from hydra.PriceLoader import load_prices

# vbt.settings.caching["blacklist"].append("Portfolio")
# vbt.settings.caching["whitelist"].extend(
#     ["Portfolio.cash_flow", "Portfolio.share_flow"]
# )
vbt.settings.ohlcv["column_names"] = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}


@njit
def aroon_entry(from_i, to_i, col, a, temp_idx_arr):
    if from_i == 0:
        w = np.where(a[:, col] > 50)[0]
        for i, num in enumerate(w):
            temp_idx_arr[i] = num

    for i in range(len(temp_idx_arr)):
        if temp_idx_arr[i] > from_i and temp_idx_arr[i] < to_i:
            return temp_idx_arr[i : i + 1]
    return temp_idx_arr[:0]


@njit
def aroon_exit(from_i, to_i, col, a, temp_idx_arr):
    if temp_idx_arr[-1] != 42:
        temp_idx_arr[-1] = 42
        w = np.where(a[:, col] < -50)[0]
        for i, num in enumerate(w):
            temp_idx_arr[i] = num

    for i in range(len(temp_idx_arr)):
        if temp_idx_arr[i] > from_i and temp_idx_arr[i] < to_i:
            return temp_idx_arr[i : i + 1]
    return temp_idx_arr[:0]


@njit
def order_func(oc, open):
    # print("OC", oc.i, oc.col)
    # print("OPEN", open, open[oc.i, oc.col])
    order = create_order_nb(size=10, price=oc.close[oc.i, oc.col])
    print(order)
    return order


def run_sim(
    pairs,
    batches,
    startDate="2020-01-01",
    endDate="2021-01-01",
    interval=1,
    path="../data/kraken",
    name=None,
):
    for pair in pairs:
        printd("Simulating", pair)
        prices = load_prices(pair, path, startDate, endDate, interval)
        # printd("Prices:", prices)
        printd(pair, "prices loaded")

        for count, batch in enumerate(batches):
            printd(pair, "batch", count, ": Generating Indicators")

            AROONOSC = vbt.IndicatorFactory.from_talib("AROONOSC")
            # printd(help(AROONOSC.run))
            aroonosc = AROONOSC.run(
                prices["high"], prices["low"], **batch.get("AROONOSC")
            )

            SAREXT = vbt.IndicatorFactory.from_talib("SAREXT")
            # printd(help(SAREXT.run))
            sarext = SAREXT.run(prices["high"], prices["low"], **batch.get("SAREXT"))
            printd(pair, "batch", count, ": Generated Indicators")

            printd(pair, "batch", count, ": Generating Strategy")
            # Build signal generator
            AroonStrategy = SignalFactory(input_names=["aroon"]).from_choice_func(
                entry_choice_func=aroon_entry,
                entry_settings=dict(
                    pass_inputs=["aroon"],
                    pass_kwargs=["temp_idx_arr"],  # built-in kwarg
                ),
                exit_choice_func=aroon_exit,
                exit_settings=dict(
                    pass_inputs=["aroon"],
                    pass_kwargs=["temp_idx_arr"],  # built-in kwarg
                ),
                # forward_flex_2d=True,
            )
            # Run strategy signal generator
            aroon_signals = AroonStrategy.run(aroonosc.real)
            printd(pair, "batch", count, ": Generated Strategy")

            printd(pair, "batch", count, ": Simulating Orders")
            # print(prices["close"], prices["open"])

            # portfolio = vbt.Portfolio.from_order_func(
            #     prices["close"],
            #     order_func,
            #     prices["open"],
            #     # prices["high"],
            #     # prices["low"],
            #     freq=f"{interval}m",
            #     init_cash=100.0,  # in $
            #     # fees = 0.0006,  # in %
            #     # slippage = 0.0025  # in %
            # )

            portfolio = vbt.Portfolio.from_signals(
                prices["close"],
                aroon_signals.entries,
                aroon_signals.exits,
                freq=f"{interval}m",
                init_cash=100.0,  # in $
                fees=0.0006,  # in %
                # slippage = 0.0025  # in %
            )

            if (
                batch["AROONOSC"]["timeperiod"][0]
                == batch["AROONOSC"]["timeperiod"][-1]
            ):
                aroonrange = batch["AROONOSC"]["timeperiod"][0]
            else:
                aroonrange = f"{batch['AROONOSC']['timeperiod'][0]}-{batch['AROONOSC']['timeperiod'][-1]}"
            # startvalue=0, offsetonreverse=0, accelerationinitlong=0.02, accelerationlong=0.02, accelerationmaxlong=0.2, accelerationinitshort=0.02, accelerationshort=0.02, accelerationmaxshort=0.2
            filename = (
                f"{pair} {startDate} {endDate} {interval} Aroon {aroonrange}.portfolio"
            )
            if name is not None:
                filename = f"{name} {filename}"

            output_dir = (
                pathlib.Path(__file__).parent.absolute().joinpath("..", "output")
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            output = output_dir.joinpath(filename)

            printd(pair, "batch", count, ": Saving file to", output)
            portfolio.save(output)
            # pickle.dump( portfolio, open( f"{filename}.p", "wb" ) )


parser = argparse.ArgumentParser("poetry run supersim")
parser.add_argument("name", type=str, nargs="?")


def start():
    t0 = time.time()
    args = parser.parse_args()
    portfolios = run_sim(
        ["XBTUSD"],
        batches=[
            {
                "AROONOSC": {"timeperiod": list(range(100, 102))},
                "SAREXT": {"startvalue": [0.01]},
            }
        ],
        startDate="2018-05-15",
        endDate="2021-05-16",
        interval=1,
        name=args.name,
    )

    t1 = time.time()
    printd("Total Time Elapsed:", t1 - t0)
    # for portfolio in portfolios:
    # printd(portfolio.total_return())
    # printd(portfolio.total_profit())
