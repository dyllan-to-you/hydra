import math
import plotly.graph_objects as go
import pandas as pd
import hydra.utils as utils
from datetime import datetime, timedelta
from hydra.DataLoader import load_prices

intervals = [1, 5, 15, 60, 720, 1440]


def gen_data_interval(data, interval):
    res = data.get(interval)
    if res is None:
        for i in intervals:
            base = data.get(i)
            if base is not None:
                res = base.resample(f"{interval}m")
                data[interval] = res
                break
    return res


def supersample_data(data, interval):
    res = {interval: data}
    for i in intervals:
        if i not in res:
            res[i] = data.resample(f"{interval}m").asfreq()
    return res


def data_resample_factory(data: "dict[int, pd.DataFrame]"):
    def resample_data(start_date, end_date, interval, size):
        fakestart = start_date - timedelta(minutes=interval * math.ceil(size / 2))
        fakeend = end_date + timedelta(minutes=interval * math.ceil(size / 2))
        interval_data = data[interval]
        precise_data = interval_data.loc[fakestart:fakeend]
        big_data = data[intervals[-1]]
        the_data = pd.concat(
            [big_data.query("index < @fakestart or index > @fakeend"), precise_data]
        ).sort_index()

        return the_data

    return resample_data


def data_extrapolator(data, figure, trace_idx, fields):
    resample = data_resample_factory(data)

    def slicer(start, end):
        size = 500
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)
        delta = end_date - start_date
        delta_m = delta.total_seconds() / 60

        interval = intervals[-1]
        for i in intervals:
            if delta_m <= size * i:
                interval = i
                break

        the_data = resample(start_date, end_date, interval, size)
        # f = open('log.txt', "a")
        # f.write(f"[{utils.now()}] {start} {end} {delta} {delta_m} {interval} \n{precise_price} \n {fakestart} {fakeend} \n{the_price.loc[fakestart:fakeend]}\n")
        # f.close()
        trace = figure.data[trace_idx]
        with figure.batch_update():
            for key, val in fields.items():
                try:
                    value = getattr(the_data, val)
                except:
                    value = the_data[val]

                setattr(trace, key, value)

    return slicer


def get_candlestick(pair, startDate, endDate):
    prices = {}
    prices[1] = load_prices(pair, startDate=startDate, endDate=endDate, interval=1)
    prices[5] = load_prices(pair, startDate=startDate, endDate=endDate, interval=5)
    prices[15] = load_prices(pair, startDate=startDate, endDate=endDate, interval=15)
    prices[60] = load_prices(pair, startDate=startDate, endDate=endDate, interval=60)
    prices[720] = load_prices(pair, startDate=startDate, endDate=endDate, interval=720)
    prices[1440] = load_prices(
        pair, startDate=startDate, endDate=endDate, interval=1440
    )

    candlestick = go.Candlestick(
        x=prices[1440].index,
        open=prices[1440]["open"],
        high=prices[1440]["high"],
        low=prices[1440]["low"],
        close=prices[1440]["close"],
        hoverinfo="all",
    )

    return candlestick, prices


# TODO: Turn into class
# TODO: move lingering variables into attributes
# TODO: Put start/end date into attribute
# TODO: Use class in `environments.ipynb` to render trendline
# TODO: Render fft aggregate
# TODO: extrapolate fft aggregate (just render min/max prediction)
# TODO: Predictive ability metrics
# TODO: Mix predictions with aroons for fun and profit
# TODO: ???
# TODO: PROFIT

data_idx = 0


def add_trace(figure: go.FigureWidget, trace, data, fields, startDate, endDate):
    global data_idx
    figure.add_trace(trace)
    update_candlestick = data_extrapolator(
        data,
        figure,
        data_idx,
        fields,
    )

    register_handler(figure, update_candlestick, startDate, endDate)
    data_idx += 1


handle_list = []


def handler(obj, xrange):
    [start, end] = xrange.range
    for fn in handle_list:
        fn(start, end)


def register_handler(figure, fn, startDate, endDate):
    fn(startDate, endDate)
    handle_list.append(fn)
    figure.layout.on_change(handler, "xaxis")


def graph(pair, startDate, endDate):
    figure = go.FigureWidget(
        data=[],
        # data=[candlestick],
        layout=go.Layout(
            title=dict(text="FLYBABYFLY"),
            barmode="overlay",
            yaxis={"fixedrange": False},
            yaxis2=dict(
                fixedrange=False,
                domain=[0, 1],
            ),
        ),
    )
    candlestick, prices = get_candlestick(pair, startDate, endDate)
    add_trace(
        figure,
        candlestick,
        prices,
        {"x": "index", "open": "open", "high": "high", "low": "low", "close": "close"},
        startDate,
        endDate,
    )

    return figure


# def add_handler(handler):


if __name__ == "__main__":
    pair = "XBTUSD"
    startDate = "2017-05-15 00:00"
    endDate = "2017-05-16 00:00"
    figure, update_candlestick = graph(pair, startDate, endDate)
    update_candlestick(startDate, endDate)

    def handler(obj, xrange):
        print("HANDLE ME")
        [start, end] = xrange.range
        update_candlestick(start, end)

    figure.layout.on_change(handler, "xaxis")
    figure.show()