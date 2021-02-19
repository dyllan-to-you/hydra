import pandas as pd
import numpy as np
from bokeh.layouts import column
from bokeh.plotting import curdoc, figure
from bokeh.io import curdoc, show
from bokeh.models import CustomJS, ColumnDataSource, HoverTool, WheelZoomTool
from bokeh.models.widgets import DateRangeSlider


def add_columns(df):
    df["Date"] = df.index.to_pydatetime()
    return df


class Chart:
    # bars_to_display - set the zoom level (more bars = smaller candles)
    def __init__(self, bars_to_display=60, interval=1):
        self.bars_to_display = bars_to_display
        self.interval = interval
        self.history = ColumnDataSource()
        self.trades = ColumnDataSource()

    def scale_y_axis(self, plot):
        return CustomJS(
            args={"y_range": plot.y_range, "source": self.history},
            code="""
            clearTimeout(window._autoscale_timeout);

            var date = source.data.date,
                low = source.data.low,
                high = source.data.high,
                start = cb_obj.start,
                end = cb_obj.end,
                min = Infinity,
                max = -Infinity;

            for (var i=0; i < date.length; ++i) {
                if (start <= date[i] && date[i] <= end) {
                    max = Math.max(high[i], max);
                    min = Math.min(low[i], min);
                }
            }
            var pad = (max - min) * .05;

            window._autoscale_timeout = setTimeout(function() {
                y_range.start = min - pad;
                y_range.end = max + pad;
            });
        """,
        )

    def update(self, start, end):
        self.update_history(start, end)
        self.update_trades(start, end)

    def update_history(self, start, end):
        """Update the data source to be displayed.
        This is called once when the plot initiates, and then every time the slider moves, or a different instrument is
        selected from the dropdown.
        """

        # create new view from dataframe
        df_view = self.history_df.loc[start:end]

        # create new source
        new_source = df_view.to_dict(orient="list")

        # add colors to be used for plotting bull and bear candles
        colors = [
            "green" if cl >= op else "red"
            for (cl, op) in zip(df_view["Close"], df_view["Open"])
        ]
        new_source["colors"] = colors

        # source.data.update(new_source)
        self.history.data = new_source

    def update_trades(self, start, end):
        # create new view from dataframe
        print(self._trade_df, start, end)
        df_view = self.trade_df.loc[start:end]

        # create new source
        new_source = df_view.to_dict(orient="list")
        self.trades.data = new_source

    def make_plot(self):
        """Draw the plot using the ColumnDataSource"""

        p = figure(
            title=f"OHLC",
            plot_height=400,
            plot_width=1200,
            x_axis_type="datetime",
            # tools="xpan,xwheel_zoom,undo,redo,reset,crosshair,save",
            # active_drag="xpan",
            # active_scroll="xwheel_zoom",
        )
        p.segment(
            "Date",
            "High",
            "Date",
            "Low",
            source=self.history,
            line_width=1,
            color="black",
        )  # plot the wicks
        p.vbar(
            "Date",
            50 * self.interval * 1000,
            "Open",
            "Close",
            source=self.history,
            line_color="colors",
            fill_color="colors",
        )  # plot the body

        # Plot Trades
        p.circle(
            x="Date", y="Price", size=20, color="color", alpha=0.5, source=self.trades
        )
        # p.circle(x="Date", y="Price", size=10, color="color", alpha=0.5)

        hover = HoverTool(
            tooltips=[
                ("Date", "@Date{%F %T}"),
                ("Open", "@Open{0.0000f}"),
                ("High", "@High{0.0000f}"),
                ("Low", "@Low{0.0000f}"),
                ("Close", "@Close{0.0000f}"),
            ],
            formatters={"@Date": "datetime"},
        )
        p.add_tools(hover)
        # p.toolbar.active_scroll = p.select_one(WheelZoomTool)

        return p

    def draw_profit(self, main):
        p = figure(
            plot_height=250,
            plot_width=1200,
            x_range=main.x_range,
            x_axis_type="datetime",
        )
        print(self.trades)
        p.line(x="Date", y="cash", color="green", source=self.trades)
        return p

    def draw_aroon(self, main, name):
        p = figure(
            plot_height=250,
            plot_width=1200,
            x_range=main.x_range,
            x_axis_type="datetime",
        )
        p.line(x="Date", y=f"{name}.down", color="red", source=self.history)
        p.line(x="Date", y=f"{name}.up", color="green", source=self.history)
        return p

    def slider_handler(self, attr, old, new):
        start, end = new
        dates = pd.to_datetime([start, end], unit="ms").tolist()
        print("Handler:", dates)
        """Handler function for the slider. Updates the ColumnDataSource to a new range given by the slider's position."""
        self.update(*dates)

    @property
    def trade_df(self):
        return self._trade_df

    @trade_df.setter
    def trade_df(self, df):
        trades = pd.json_normalize(df, sep=".")
        buys = pd.DataFrame(trades["buy"].tolist(), columns=["Date", "Price"])
        buys["color"] = "green"
        buys = buys.join(trades[["cash_buy"]]).rename(columns={"cash_buy": "cash"})
        # buys.set_index("Date", inplace=True)
        sells = pd.DataFrame(trades["sell"].tolist(), columns=["Date", "Price"])
        sells["color"] = "red"
        sells = sells.join(trades[["cash_sell"]]).rename(columns={"cash_sell": "cash"})
        self._trade_df = buys.append(sells)
        print(trades, "\n", sells, "\n", buys, "\n", self._trade_df)
        self._trade_df["Date"] = pd.to_datetime(self._trade_df["Date"], unit="s")
        self._trade_df.set_index("Date", inplace=True)
        self._trade_df.sort_index(inplace=True)
        self._trade_df.dropna(inplace=True)
        add_columns(self._trade_df)

    @property
    def history_df(self):
        return self._history_df

    @history_df.setter
    def history_df(self, df):
        self._history_df = add_columns(df)

    def graph(self, sim):
        # output_file(filename)
        # get data based on selection

        self.sim = sim
        self.history_df = sim.hydra.price_history_df
        self.trade_df = sim.trade_history

        start = self.history_df["Date"][0]
        end = self.history_df["Date"][-1]
        width = pd.Timedelta(minutes=self.interval * self.bars_to_display)
        slider = DateRangeSlider(
            title="Date Range: ",
            start=start,
            end=end,
            value=(
                start,
                start + width,
            ),
            step=self.interval,
        )

        slider.on_change("value_throttled", self.slider_handler)

        # initialize the history
        self.update(start, start + width)

        # draw the plot
        plot = self.make_plot()
        # p.x_range.js_on_change("start", callback)

        slider.js_on_change("value", self.scale_y_axis(plot))
        aroon = self.draw_aroon(plot, sim.hydra.strategy.indicator.name)
        profit = self.draw_profit(plot)

        curdoc().add_root(
            column(plot, aroon, profit, slider, sizing_mode="stretch_width")
        )
        # show(column(plot, slider))
