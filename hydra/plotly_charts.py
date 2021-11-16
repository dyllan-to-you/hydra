import math
import plotly.graph_objects as go
import matplotlib
import pandas as pd
from plotly.subplots import make_subplots
import hydra.utils as utils
from datetime import datetime, timedelta
from dataloader import load_prices

intervals = [1, 5, 15, 60, 720, 1440]


def matplotlib_to_plotly(cmap_name, pl_entries):
    cmap = matplotlib.cm.get_cmap(cmap_name)

    h = 1.0 / (pl_entries - 1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = map(np.uint8, np.array(cmap(k * h)[:3]) * 255)
        pl_colorscale.append([k * h, "rgb" + str((C[0], C[1], C[2]))])

    return pl_colorscale


def supersample_data(
    data, interval, useIntervals=[], alwaysUseData=False, approximate=False
):
    res = {interval: data}
    for i in intervals:
        if i not in res:
            if alwaysUseData:
                res[i] = data if i in useIntervals else data.truncate(copy=True)
            else:
                if approximate:
                    res[i] = (
                        data.round(f"{i}min").avg()
                        if i in useIntervals
                        else data.truncate(copy=True)
                    )
                else:
                    raise Error("Bad resampling logic")
                    res[i] = (
                        data.resample(f"{i}min").asfreq()
                        if i in useIntervals
                        else data.truncate(copy=True)
                    )
            # print("res", i, f"{interval}T", res[i])
    return res


def data_resample_factory(data: "dict[int, pd.DataFrame]", onlySlice=False):
    def resample_data(start_date, end_date, interval):
        window_size = end_date - start_date
        fakestart = start_date - window_size
        fakeend = end_date + window_size
        interval_data = data[interval]
        precise_data = interval_data.loc[fakestart:fakeend]
        if onlySlice:
            return precise_data
        else:
            big_data = data[intervals[-1]]
            the_data = pd.concat(
                [big_data.loc[:fakestart], big_data.loc[fakeend:], precise_data]
            ).sort_index()

        return the_data

    return resample_data


def get_price_trace(pair, startDate, endDate):
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
        name="Prices",
    )

    return candlestick, prices


# DONE: Turn into class
# DONE: move lingering variables into attributes
# DONE: Put start/end date into attribute
# DONE: Use class in `environments.ipynb` to render trendline
# DONE: Render fft aggregate
# DONE: Replace complexity chart with heatmap
"""
TODO: extrapolate fft aggregate (just render min/max prediction)
- Take IFFT
- detrend
- get last min/max value & index
    - take abs value
    - reverse
    - get index of max value
    - real_index = len - index
    - check detrended_ifft[real_index] to determine whether it was a min or a max
- get largest amplitude from original 
- alternate adding/subtracting the amplitude (min/max value?) 
    every frequency/2 points after the last min/max 
- Retrend
- Return

"""
# TODO: Predictive ability metrics
# TODO: Mix predictions with aroons for fun and profit
# TODO: ???
# TODO: PROFIT


class PlotlyPriceChart:
    def __init__(self, pair, startDate, endDate):
        self.data_idx = 0
        self.handle_list = []
        self.handler = self.handler_factory()
        self.figure: go.FigureWidget = None
        self.startDate = startDate
        self.endDate = endDate
        self.generate_figure(pair)

    def slicer(self, start, end):
        size = 250
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)
        delta = end_date - start_date
        delta_m = delta.total_seconds() / 60

        interval = intervals[-1]
        for i in intervals:
            if delta_m <= size * i:
                interval = i
                break

        # f = open('log.txt', "a")
        # f.write(f"[{utils.now()}] {start} {end} {delta} {delta_m} {interval} \n{precise_price} \n {fakestart} {fakeend} \n{the_price.loc[fakestart:fakeend]}\n")
        # f.close()
        with self.figure.batch_update():
            for resample, trace_idx, fields in self.handle_list:
                the_data = resample(start_date, end_date, interval)
                trace = self.figure.data[trace_idx]
                for key, val in fields.items():
                    try:
                        value = getattr(the_data, val)
                    except:
                        value = the_data[val]

                    setattr(trace, key, value)

    def add_trace(
        self,
        trace,
        data=None,
        fields={"x": "index", "y": "values"},
        loc=(2, 1),
        traceArgs={},
        onlySlice=False,
        **kwargs,
    ):
        if "log" in kwargs:
            print("add_trace", trace, data, fields, loc)
        row, col = loc
        self.figure.add_trace(trace, row=row, col=col, **traceArgs)
        if data is not None and trace is not None:
            self.register_handler(
                (data_resample_factory(data, onlySlice), self.data_idx, fields)
            )
        self.data_idx += 1

    def handler_factory(self):
        def handler(obj, xrange):
            [start, end] = xrange.range
            self.slicer(start, end)

        return handler

    def register_handler(self, data):
        self.handle_list.append(data)
        self.slicer(self.startDate, self.endDate)
        self.figure.layout.on_change(self.handler, "xaxis")

    def generate_figure(self, pair):
        self.figure = go.FigureWidget(make_subplots(rows=2, cols=1, shared_xaxes=True))

        candlestick, prices = get_price_trace(pair, self.startDate, self.endDate)
        self.add_trace(
            candlestick,
            prices,
            {
                "x": "index",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
            },
        )

        return self

    def render(self, **kwargs):
        self.figure.update_layout(
            go.Layout(
                title=dict(text="FLYBABYFLY"),
                barmode="overlay",
                yaxis2={"fixedrange": False},
                height=800,
                # yaxis2=dict(
                #     fixedrange=False,
                #     domain=[0, 1],
                # ),
            ),
            **kwargs,
        )
        return self.figure


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