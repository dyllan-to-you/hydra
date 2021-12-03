import pathlib
import pickle
from datetime import datetime

import numpy as np
from numpy.core.numeric import cross
import pandas as pd
from pandas._testing import assert_frame_equal

import pyarrow
from dataloader import load_prices
from dataloader.binance_data import update_data

from hydra.Environments import gen_tasks
from hydra.Environments import main as bulk_analysis
from hydra.Environments import run_parallel

ROOT_WINDOW = 2 * 1440
SKIP_SAVE = False
ABORT_ON_DUPLICATE = False

RUN_SAVE_TEST = False


def main(
    startDate=None,
    runEndDatetime=None,
    pair_binance="BTCUSD",
    overlap=0.75,
    forceNewFile=True,
    config="config",
    rootWindow=ROOT_WINDOW,
):
    rootWindow_td = pd.to_timedelta(rootWindow, unit="min")
    update_data(pair_binance=pair_binance)
    output_dir = pathlib.Path("./output/enviro")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_file = output_dir / config

    if startDate is None:
        prices = load_prices("BTCUSD")
        startDate = (prices.index[0] + pd.to_timedelta("1439min")).floor("D")
    else:
        startDate = pd.to_datetime(startDate)

    # year/month/day/window.parq

    if runEndDatetime is None:
        # Subtract 1 min to save "yesterday" for the live run
        runEndDatetime = pd.to_datetime("today").floor("D") - pd.to_timedelta("1min")
    else:
        runEndDatetime = pd.to_datetime(runEndDatetime)

    print("runEndDatetime", runEndDatetime)

    # Load previous runs if exists
    try:
        accumulated_parent_data = load_accumulated_parents(output_dir)

        last_run = accumulated_parent_data.loc[
            accumulated_parent_data["parent_window_original"] == rootWindow
        ]["parent_endDate"].max()
        # If last run before current runEndDatetime, catch up to now
        if last_run < runEndDatetime.date():
            time_since_last_run = runEndDatetime - last_run

            startDate = last_run + pd.to_timedelta("1d") - rootWindow_td
            print(f"Last Run: {last_run} {time_since_last_run} ago")
            result_data, result_charts, result_aggregates = bulk_analysis(
                startDate,
                end=runEndDatetime,
                window=f"{rootWindow}min",
                detrend=True,
                pair=pair_binance,
                midnightLock=True,
            )
            catchup_parent_data: pd.DataFrame = save_results(
                result_data, result_charts, rootWindow=rootWindow, output_dir=output_dir
            )
            catchup_parent_data = preprocess_child_runs(
                overlap,
                rootWindow,
                runEndDatetime,
                catchup_parent_data,
                all_children=True,
            )
            accumulated_parent_data = accumulated_parent_data.append(
                catchup_parent_data
            )
    except (FileNotFoundError, OSError):
        print("! Could not find previous runs.")
        print(
            f"Generating First Run {startDate} - {runEndDatetime}",
        )
        result_data, result_charts, result_aggregates = bulk_analysis(
            startDate,
            end=runEndDatetime,
            window=f"{rootWindow}min",
            detrend=True,
            pair=pair_binance,
            midnightLock=True,
        )
        accumulated_parent_data: pd.DataFrame = save_results(
            result_data, result_charts, output_dir=output_dir
        )
        accumulated_parent_data = preprocess_child_runs(
            overlap,
            rootWindow,
            runEndDatetime,
            accumulated_parent_data,
            all_children=True,
        )

    # history = set(accumulated_parent_data.itertuples(index=False, name=None))
    while not accumulated_parent_data["saved"].all():
        save_accumulated_parents(output_dir, accumulated_parent_data)
        print("\nGenerating tasks for next run")
        # "startDate", "endDate", "window", "minPerCycle"
        next_runs = accumulated_parent_data.loc[
            accumulated_parent_data["saved"] == False
        ]
        print("next run", next_runs)
        tasks = [
            (
                task,
                (
                    parent_startDate,
                    parent_endDate,
                    parent_window_original,
                    parent_window,
                    child_minPerCycle,
                    child_window,
                    child_startDate,
                    child_endDate,
                    saved,
                    *task,
                ),
            )
            for (
                parent_startDate,
                parent_endDate,
                parent_window_original,
                parent_window,
                child_minPerCycle,
                child_window,
                child_startDate,
                child_endDate,
                saved,
            ) in next_runs.itertuples(index=None, name=None)
            for task in list(
                gen_tasks(
                    child_startDate,
                    end=child_endDate,
                    window=child_window,
                    overlap=overlap,
                    pair=pair_binance,
                    tuple=True,
                )
            )
        ]
        if len(tasks) == 0:
            print("No more tasks!")
            break

        tasks, taskParents = list(zip(*tasks))
        print("Checking for duplicates")
        task_set = set(tasks)
        if len(tasks) != len(task_set):
            dupeCount = len(tasks) - len(task_set)
            if ABORT_ON_DUPLICATE:
                tasks_df = pd.DataFrame(
                    taskParents,
                    columns=[
                        "parent_startDate",
                        "parent_endDate",
                        "parent_window_original",
                        "parent_window",
                        "child_minPerCycle",
                        "child_window",
                        "child_startDate",
                        "child_endDate",
                        "saved",
                        "pair",
                        "startDate",
                        "window",
                        "window_original",
                    ],
                )
                dupes = tasks_df[
                    tasks_df.duplicated(
                        keep=False, subset=["pair", "startDate", "window"]
                    )
                ]
                raise Exception(f"{len(dupes)} Duplicates found: {dupes}")
            print(f"{dupeCount} Duplicates filtered")
            tasks = task_set

        # tasks_df = pd.DataFrame(
        #     tasks, columns=["pair", "startDate", "window", "window_original"]
        # )
        # tasks_df = pd.DataFrame(
        #     taskParents,
        #     columns=[
        #         "parent_startDate",
        #         "parent_endDate",
        #         "parent_window_original",
        #         "parent_window",
        #         "child_minPerCycle",
        #         "child_window",
        #         "child_startDate",
        #         "child_endDate",
        #         "saved",
        #         "pair",
        #         "startDate",
        #         "window",
        #         "window_original",
        #     ],
        # )
        # tasks_df["window_original_m"] = tasks_df["window_original"] / pd.to_timedelta(
        #     "1min"
        # )
        # print("=== TASKS ==", tasks_df)
        # print(
        #     tasks_df.loc[
        #         (
        #             (tasks_df["window_original_m"] >= 8)
        #             & (tasks_df["window_original_m"] < 13)
        #         )
        #     ]
        # )

        print(f"Executing {len(tasks)} Tasks")
        results = [result for result in run_parallel(tasks) if result is not None]
        data, charts = zip(*results)
        child_runs = save_results(data, charts, output_dir=output_dir)
        accumulated_parent_data["saved"] = True

        child_runs = preprocess_child_runs(
            overlap, rootWindow, runEndDatetime, child_runs
        )
        accumulated_parent_data = accumulated_parent_data.append(child_runs)

    print("All layers run")
    accumulated_parent_data["saved"] = True
    save_accumulated_parents(output_dir, accumulated_parent_data)

    """
                                                   parent_startDate    parent_endDate  parent_window  child_minPerCycle
figname                                                                                             
(D)BTCUSD 2013-10-07 00:01 - 2013-10-08 00:0014... 2013-10-07 00:01:00 2013-10-08    1440      14.39
(D)BTCUSD 2013-10-08 00:01 - 2013-10-09 00:0014... 2013-10-08 00:01:00 2013-10-09    1440      14.39
(D)BTCUSD 2013-10-09 00:01 - 2013-10-10 00:0014... 2013-10-09 00:01:00 2013-10-10    1440      14.39
(D)BTCUSD 2013-10-10 00:01 - 2013-10-11 00:0014... 2013-10-10 00:01:00 2013-10-11    1440    1440.00
(D)BTCUSD 2013-10-10 00:01 - 2013-10-11 00:0014... 2013-10-10 00:01:00 2013-10-11    1440      14.39
...                                                                ...        ...     ...        ...
(D)BTCUSD 2021-11-22 00:01 - 2021-11-23 00:0014... 2021-11-22 00:01:00 2021-11-23    1440      14.39
(D)BTCUSD 2021-11-23 00:01 - 2021-11-24 00:0014... 2021-11-23 00:01:00 2021-11-24    1440    1440.00
(D)BTCUSD 2021-11-23 00:01 - 2021-11-24 00:0014... 2021-11-23 00:01:00 2021-11-24    1440      14.39
(D)BTCUSD 2021-11-24 00:01 - 2021-11-25 00:0014... 2021-11-24 00:01:00 2021-11-25    1440    1440.00
(D)BTCUSD 2021-11-24 00:01 - 2021-11-25 00:0014... 2021-11-24 00:01:00 2021-11-25    1440      14.39
    """
    return


def save_accumulated_parents(
    output_dir, accumulated_parent_data, filename="accumulated_parents"
):
    path = output_dir / f"{filename}.parq"
    print(f"Saving Accumulated Parent Data to {path}")
    # print(accumulated_parent_data)
    if not SKIP_SAVE:
        try:
            accumulated_parent_data_tosave = accumulated_parent_data.copy()
            accumulated_parent_data_tosave[
                "child_window"
            ] = accumulated_parent_data_tosave["child_window"] / pd.to_timedelta("60s")

            accumulated_parent_data_tosave.to_parquet(path)

            if RUN_SAVE_TEST:
                loaded = load_accumulated_parents(output_dir, filename)
                print("a", accumulated_parent_data)
                print(loaded)
                # print(f"SAVE TEST! \n{accumulated_parent_data.compare(loaded)}")
                assert_frame_equal(accumulated_parent_data, loaded)
                # assert accumulated_parent_data.equals(
                #     loaded
                # ), f"SAVE TEST FAILED! \n{accumulated_parent_data=} \n{loaded=}"

        except pyarrow.ArrowNotImplementedError as e:
            print(accumulated_parent_data.dtypes)
            raise e


def load_accumulated_parents(output_dir, filename="accumulated_parents"):
    print(f"Loading previous runs from {filename}")
    accumulated_parents = pd.read_parquet(output_dir / f"{filename}.parq")
    accumulated_parents["child_window"] = pd.to_timedelta(
        accumulated_parents["child_window"], unit="min"
    )

    # print(accumulated_parents)
    return accumulated_parents


def partition_filename_factory(window, prefix="", suffix=""):
    def cb(keys):
        # year, month, day = keys
        return f"{prefix}{window}{suffix}.parq"

    return cb


def save_results(result_data, result_charts, rootWindow=ROOT_WINDOW, output_dir=None):
    # if isinstance(window, str):
    #     window_minutes = pd.to_timedelta(window).total_seconds() / 60
    # else:
    #     window_minutes = window

    result_data_df = save_result_data(output_dir, result_data, rootWindow=rootWindow)
    extra_data_df = save_extra_data(output_dir, result_charts, rootWindow=rootWindow)

    accumulated_parent_data: pd.DataFrame = result_data_df[
        ["startDate", "endDate", "window_original", "window", "minPerCycle"]
    ]
    accumulated_parent_data = accumulated_parent_data.rename(
        columns={
            "startDate": "parent_startDate",
            "endDate": "parent_endDate",
            "window_original": "parent_window_original",
            "window": "parent_window",
            "minPerCycle": "child_minPerCycle",
        }
    )

    return accumulated_parent_data


def preprocess_child_runs(
    overlap,
    rootWindow,
    runEndDatetime,
    accumulated_parent_data: pd.DataFrame,
    all_children=False,
):

    # remove minPerCycle < 15 for irrelevance, incl 0 aka. trend-based prediction
    accumulated_parent_data = accumulated_parent_data.loc[
        accumulated_parent_data["child_minPerCycle"] >= 15
    ]

    # remove children that have the same window as parent
    accumulated_parent_data = accumulated_parent_data.loc[
        ~np.isclose(
            np.floor(accumulated_parent_data["parent_window"]),
            accumulated_parent_data["child_minPerCycle"],
        )
    ]

    if not all_children:
        # Only use child with largest window
        max_minPerCycle = accumulated_parent_data.groupby(
            ["parent_endDate", "parent_window_original"], sort=False
        )["child_minPerCycle"].transform(max)
        max_minPerCycle_bool = (
            accumulated_parent_data["child_minPerCycle"] == max_minPerCycle
        )

        accumulated_parent_data = accumulated_parent_data[max_minPerCycle_bool]

    """
    child minpercycle * (parent window original / parent window) * (floor / parent window) = child window

    
    """

    # child_minPerCycle is only inaccurate in cases
    print(
        accumulated_parent_data["parent_window_original"],
        accumulated_parent_data["parent_window"],
    )
    accumulated_parent_data["child_window"] = pd.to_timedelta(
        accumulated_parent_data["child_minPerCycle"]
        * (
            accumulated_parent_data["parent_window_original"]
            / accumulated_parent_data["parent_window"]
        )
        * (
            accumulated_parent_data["parent_window"]
            / np.floor(accumulated_parent_data["parent_window"])
        ),
        unit="min",
    )
    print(accumulated_parent_data)
    # ROOT NUMBER DEBUGGING
    # rootCalc = accumulated_parent_data.loc[
    #     :,
    #     [
    #         "parent_window_original",
    #         "parent_window",
    #         "child_minPerCycle",
    #         "child_window",
    #     ],
    # ]

    # rootCalc["rootNumber"] = np.around(
    #     rootWindow / rootCalc["parent_window_original"], 2
    # )

    # assert (np.isclose(rootCalc["rootNumber"] % 1, 0)).all(), f"{rootCalc}"
    # print(
    #     rootCalc.loc[(rootCalc["window_original"] > 50) and (rootCalc["window"] < 55)]
    # )
    # raise Exception("watwatwat")

    accumulated_parent_data["child_startDate"] = (
        accumulated_parent_data["parent_endDate"]
        - accumulated_parent_data["child_window"]
        + pd.to_timedelta(1, unit="min")
    ).round("1min")

    # END DATE
    accumulated_parent_data["child_windowIncrement"] = np.where(
        accumulated_parent_data["parent_window_original"] == rootWindow,
        1440,  # if parent is root, assume 1d overlap
        accumulated_parent_data["parent_window_original"] * (1 - overlap),
    )

    accumulated_parent_data.loc[
        accumulated_parent_data["child_windowIncrement"] < 1,
        "child_windowIncrement",
    ] = 1
    accumulated_parent_data["child_windowIncrement"] = accumulated_parent_data[
        "child_windowIncrement"
    ].round()

    accumulated_parent_data["child_endDate"] = accumulated_parent_data[
        "parent_endDate"
    ] + pd.to_timedelta(accumulated_parent_data["child_windowIncrement"], unit="min")

    accumulated_parent_data = accumulated_parent_data.drop(
        columns="child_windowIncrement"
    )

    accumulated_parent_data.loc[
        accumulated_parent_data["child_endDate"] > runEndDatetime, "child_endDate"
    ] = runEndDatetime

    accumulated_parent_data["saved"] = False

    return accumulated_parent_data


def save_result_data(
    output_dir,
    result_data,
    rootWindow=ROOT_WINDOW,
):
    print("Saving result data")
    df = pd.concat(result_data)

    # Parquet preprocessing
    ## extrapolations
    df["ifft_extrapolated_wavelength"] = df[
        "ifft_extrapolated_wavelength"
    ] / pd.Timedelta("60s")
    df["first_extrapolated_date"] = df["first_extrapolated_date"].astype(
        "datetime64[ms]"
    )
    df["startDate"] = df["startDate"].astype("datetime64[ms]")
    df["endDate"] = df["endDate"].astype("datetime64[ms]")

    # Initialize Keys
    df["year"] = df["endDate"].dt.year.astype(int)
    df["month"] = df["endDate"].dt.month.astype(int)
    df["day"] = df["endDate"].dt.day.astype(int)
    df["window"] = df["window"] / pd.Timedelta("60s")
    df["window_original"] = df["window_original"] / pd.Timedelta("60s")

    df["rootNumber"] = np.around(rootWindow / df["window_original"], 2)

    assert (
        np.isclose(df["rootNumber"] % 1, 0)
    ).all(), f"{df.loc[~np.isclose(df['rootNumber'] % 1, 0), ['window', 'window_original', 'rootNumber']]}"

    df["rootNumber"] = df["rootNumber"].astype(int)
    # print("RESULTS", df)

    if not SKIP_SAVE:
        try:
            for name, group in df.groupby("rootNumber"):
                group.to_parquet(
                    output_dir,
                    partition_cols=["year", "month", "day"],
                    partition_filename_cb=partition_filename_factory(name),
                    allow_truncated_timestamps=True,
                )
        except pyarrow.ArrowNotImplementedError as e:
            print(df.dtypes)
            raise e
        except pyarrow.ArrowInvalid as e:
            print(df.dtypes)
            raise e

    return df


def save_extra_data(
    output_dir,
    result_charts,
    rootWindow=ROOT_WINDOW,
):
    print("Saving extra data")

    data = [
        {
            "index": c["figname"],
            "startDate": c["startDate"],
            "endDate": c["endDate"],
            "window": c["window"],
            "window_original": c["window_original"],
            "significant_dates": c[
                "significant_extrapolation_line"
            ].index.to_pydatetime(),
            "significant_values": c["significant_extrapolation_line"].values,
            "insignificant_dates": c[
                "insignificant_extrapolation_line"
            ].index.to_pydatetime(),
            "insignificant_values": c["insignificant_extrapolation_line"].values,
        }
        for c in result_charts
    ]
    df = pd.DataFrame(data).set_index("index")

    df["startDate"] = df["startDate"].astype("datetime64[ms]")
    df["endDate"] = df["endDate"].astype("datetime64[ms]")

    # Initialize Keys
    df["year"] = df["endDate"].dt.year.astype(int)
    df["month"] = df["endDate"].dt.month.astype(int)
    df["day"] = df["endDate"].dt.day.astype(int)
    df["window"] = df["window"] / pd.Timedelta("60s")
    df["window_original"] = df["window_original"] / pd.Timedelta("60s")

    df["rootNumber"] = np.around(rootWindow / df["window_original"], 2)

    assert (
        np.isclose(df["rootNumber"] % 1, 0)
    ).all(), f"{df.loc[~np.isclose(df['rootNumber'] % 1, 0), ['window', 'window_original', 'rootNumber']]}"

    df["rootNumber"] = df["rootNumber"].astype(int)

    # print("CHART", df)
    # print(df.dtypes)
    # return df
    if not SKIP_SAVE:
        try:
            for name, group in df.groupby("rootNumber"):
                group.to_parquet(
                    output_dir,
                    partition_cols=["year", "month", "day"],
                    partition_filename_cb=partition_filename_factory(
                        name, suffix=".xtrp"
                    ),
                    allow_truncated_timestamps=True,
                )
        except pyarrow.ArrowNotImplementedError as e:
            print(df.dtypes)
            raise e
        except pyarrow.ArrowInvalid as e:
            print(df.dtypes)
            raise e
    return df


if __name__ == "__main__":
    startDate = (pd.to_datetime("today") - pd.to_timedelta("7 days") * 52).floor("D")
    endDate = (
        pd.to_datetime("today").floor("D")
        - pd.to_timedelta("1 days")
        - pd.to_timedelta("1min")
    )

    # startDate = None
    endDate = None
    main(startDate=startDate, runEndDatetime=endDate)


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
