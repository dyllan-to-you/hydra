import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def graph(sim):
    df = sim.hydra.price_history_df
    trades = pd.json_normalize(sim.trade_history, sep=".")
    buys = pd.DataFrame(trades["buy"].tolist(), columns=["Date", "Price"])
    # buys.set_index("Date", inplace=True)
    sells = pd.DataFrame(trades["sell"].tolist(), columns=["Date", "Price"])
    # sells.set_index("Date", inplace=True)

    aup = f"{sim.hydra.strategy.indicator.name}.up"
    adown = f"{sim.hydra.strategy.indicator.name}.down"
    hovertext = []
    for i in range(len(df["Open"])):
        hovertext.append(
            f"AroonUp: {str(df[aup][i])} <br>AroonDown: {str(df[adown][i])}"
        )

    fig = make_subplots(rows=2, cols=1, start_cell="top-left", shared_xaxes=True)
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            customdata=df[
                [
                    f"{sim.hydra.strategy.indicator.name}.up",
                    f"{sim.hydra.strategy.indicator.name}.down",
                ]
            ],
            # text=hovertext,
            hoverinfo="all",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            name="Buys",
            marker_color="green",
            x=buys["Date"],
            y=buys["Price"],
            text=buys["Price"],
            mode="markers",
            marker_line_width=2,
            marker_size=10,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            name="Sells",
            marker_color="red",
            x=sells["Date"],
            y=sells["Price"],
            text=sells["Price"],
            mode="markers",
            marker_line_width=2,
            marker_size=10,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            name="Aroon Up",
            marker_color="green",
            x=df.index,
            y=df[f"{sim.hydra.strategy.indicator.name}.up"],
            mode="lines",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            name="Aroon Down",
            marker_color="red",
            x=df.index,
            y=df[f"{sim.hydra.strategy.indicator.name}.down"],
            mode="lines",
        ),
        row=1,
        col=1,
    )
    fig.update_layout(title=sim.hydra.name, xaxis2_rangeslider_visible=False)

    return fig