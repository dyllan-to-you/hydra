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
    data = data.sort_index()
    res = {interval: data}
    for i in intervals:
        if i not in res:
            if alwaysUseData:
                res[i] = data if i in useIntervals else data.truncate(copy=True)
            else:
                if approximate:
                    print("data", data)  # , data.round(f"{i}min").avg())
                    print(data.dtypes)
                    res[i] = (
                        data.round(f"{i}min").avg()
                        if i in useIntervals
                        else data.truncate(copy=True)
                    )
                else:
                    raise Exception("Bad resampling logic")
                    res[i] = (
                        data.resample(f"{i}min").asfreq()
                        if i in useIntervals
                        else data.truncate(copy=True)
                    )
            # print("res", i, f"{interval}T", res[i])
    return res


def data_resample_factory(
    data: "dict[int, pd.DataFrame]", onlySlice=False, metaHandler=None
):
    _data = data

    def resample_data(start_date, end_date, interval, traceIdx, figure, meta={}):
        nonlocal _data

        # window_size = end_date - start_date
        # fakestart = start_date - window_size
        # fakeend = end_date + window_size
        utils.write(f"resampling {traceIdx} {meta}")
        if metaHandler is not None and meta:
            _data = metaHandler(_data, figure, meta)

        interval_data = _data[interval]
        try:
            precise_data = interval_data.loc[start_date:end_date]
            if onlySlice:
                return precise_data
            else:
                big_data = _data[intervals[-1]]
                the_data = pd.concat(
                    [big_data.loc[:start_date], big_data.loc[end_date:], precise_data]
                ).sort_index()

            return the_data
        except Exception as e:
            print(interval_data.index, start_date, end_date)
            raise e

    return resample_data


def resample_prices(prices: pd.DataFrame, base_interval=1, desired_interval=60):
    if desired_interval < base_interval:
        raise Exception(
            f"Cannot resample {base_interval}m data into {desired_interval}m"
        )
    return prices.resample(f"{desired_interval}min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )


def get_price_trace(pair, startDate, endDate):
    prices = {}
    prices[1] = load_prices(pair, startDate=startDate, endDate=endDate, interval=1)
    prices[5] = resample_prices(prices[1], base_interval=1, desired_interval=5)
    prices[15] = resample_prices(prices[1], base_interval=1, desired_interval=15)
    prices[60] = resample_prices(prices[1], base_interval=1, desired_interval=60)
    prices[720] = resample_prices(prices[1], base_interval=1, desired_interval=720)
    prices[1440] = resample_prices(prices[1], base_interval=1, desired_interval=1440)

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
    def __init__(self, pair, startDate, endDate, loc):
        self.data_idx = 0
        self.meta = {"timeSlider": 50}
        self.handle_list = []
        self.handler = self.handler_factory()
        self.slider_handler = self.slider_handler_factory()
        self.slider_meta_handler = self.slider_meta_handler_factory()
        self.figure: go.FigureWidget = None
        self.startDate = startDate
        self.endDate = endDate
        self.xrange = [startDate, endDate]
        self.generate_figure(pair, loc=loc)

    def slicer(self, start, end):
        # self.figure.update_layout(title="slicer" + str(self.meta))
        timeSlider = self.meta.get("timeSlider", None)

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

        # self.figure.update_layout(title="slicer halfway" + str(self.meta))
        # utils.write(f"[{utils.now()}] {start} {end} {delta} {delta_m} {interval} \n{precise_price} \n {fakestart} {fakeend} \n{the_price.loc[fakestart:fakeend]}\n")
        with self.figure.batch_update():
            # self.figure.update_layout(title="slicer batshit" + str(self.meta))
            for resample, trace_idx, fields in self.handle_list:
                the_data = resample(
                    start_date, end_date, interval, trace_idx, self.figure, self.meta
                )

                # slider_handler
                # if self.figure.layout.title.text is not None:
                #     slider_input = int(self.figure.layout.title.text[0:2])

                if timeSlider is not None:
                    slider_input = timeSlider

                    if slider_input > 0:
                        sliderEnd = start_date + (delta * slider_input / 100)
                        if trace_idx > 0:
                            the_data = the_data.loc[the_data["endDate"] <= sliderEnd]
                        else:
                            the_data = the_data.loc[the_data.index <= sliderEnd]

                trace = self.figure.data[trace_idx]
                for key, val in fields.items():
                    try:
                        value = getattr(the_data, val)
                    except:
                        value = the_data[val]

                    setattr(trace, key, value)

            # self.figure.update_layout(
            #     title="slicer cray" + str(self.meta) + str(timeSlider)
            # )

    def add_trace(
        self,
        trace,
        data=None,
        fields={"x": "index", "y": "values"},
        loc=(2, 1),
        traceArgs={},
        onlySlice=False,
        metaHandler=None,
        **kwargs,
    ):
        if "log" in kwargs:
            print("add_trace", trace, data, fields, loc)
        row, col = loc
        self.figure.add_trace(trace, row=row, col=col, **traceArgs)
        if data is not None and trace is not None:
            print("registering handler", self.data_idx, metaHandler is not None)
            self.register_handler(
                (
                    data_resample_factory(data, onlySlice, metaHandler),
                    self.data_idx,
                    fields,
                )
            )
        self.data_idx += 1

    def handler_factory(self):
        def handler(obj, xrange):
            [start, end] = xrange.range
            self.xrange = [start, end]
            self.slicer(start, end)

        return handler

    def slider_handler_factory(self):
        def handler(obj, title):
            # utils.write("slidercall", title)
            slider_input = int(title.text[0:2])
            self.meta["timeSlider"] = slider_input
            [start, end] = self.xrange
            self.slicer(start, end)

        return handler

    def slider_meta_handler_factory(self):
        def handler(obj, meta):
            # if self.meta.timeSlider != meta.timeSlider:
            self.meta = {**self.meta, **meta}
            # utils.write("metacall", self.meta)
            self.figure.update_layout(title=str(self.meta))

            [start, end] = self.xrange
            self.slicer(start, end)

        return handler

    def register_handler(self, data):
        self.handle_list.append(data)
        self.slicer(self.startDate, self.endDate)
        self.figure.layout.on_change(self.handler, "xaxis")
        # self.figure.layout.on_change(self.slider_handler, "title")
        self.figure.layout.on_change(self.slider_meta_handler, "meta")

    def generate_figure(self, pair, loc):
        rows, cols = loc
        self.figure = go.FigureWidget(
            make_subplots(rows=rows, cols=cols, shared_xaxes=True)
        )

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
            loc=loc,
        )

        return self

    def render(self, zoomStart, zoomEnd, **kwargs):
        self.figure.update_layout(
            go.Layout(
                title=f"Charts!",
                barmode="overlay",
                yaxis={"fixedrange": False},
                yaxis2={"fixedrange": False},
                height=800,
                xaxis=dict(range=[zoomStart, zoomEnd]),
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
