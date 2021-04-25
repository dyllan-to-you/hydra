from enum import Enum
import gc
from typing import Dict, List, NamedTuple, OrderedDict
import weakref
from numpy import number
import pandas
from hydra.SuperSim import load_prices
import sqlite3
from datetime import datetime, timedelta
from tqdm import tqdm


def str_to_datetime(str) -> datetime:
    return (
        datetime.strptime(str, "%Y-%m-%d %H:%M:%S")
        if not isinstance(str, datetime)
        else str
    )


def get_aroon_id(aroon):
    timestamp, interval, timeperiod, threshold = aroon
    return f"{interval}-{timeperiod}-{threshold}"


def get_simulation_id(entry, exit):
    return f"{get_aroon_id(entry)}:{get_aroon_id(exit)}"


class Direction(Enum):
    BUY = True
    SELL = False


class Order(NamedTuple):
    timestamp: datetime
    order: Direction


class Trade(NamedTuple):
    buy_time: datetime
    profit: number


class Simulation:
    buy_time: datetime  # timestamp | None
    history: List[Trade]

    def __init__(self, buy, trade):
        self.open_position(buy)
        self.history = [trade]

    def has_open_position(self):
        return self.buy_time is not None

    def open_position(self, time):
        self.buy_time = str_to_datetime(time) if time is not None else None

    def close_position(self, profit):
        self.history.append(Trade(self.buy_time, profit))
        self.buy_time = None

    def trim_history(self, time):
        if len(self.history):
            buy_time, profit = self.history[0]
            if buy_time == time:
                self.history.pop(0)


class Aroon(NamedTuple):
    timestamp: datetime
    interval: int
    timeperiod: int
    threshold: int


class Price(NamedTuple):
    timestamp: pandas.Timestamp
    open: number
    high: number
    low: number
    close: number
    volume: number


class Context:
    current_time: datetime
    prices: pandas.DataFrame
    db: sqlite3.Connection
    entries: List
    simulations: Dict[str, Simulation]
    sims_by_entry: Dict[str, Simulation]
    delta: timedelta
    fee: number
    orders: List[Order]
    ideal_exit: str
    window_prices: OrderedDict[datetime, Price]

    def __init__(self, prices, db, delta, fee):
        self.prices = prices
        self.db = db
        self.entries = list()
        self.simulations = dict()
        self.sims_by_entry = dict()
        self.delta = timedelta(minutes=delta)
        self.fee = fee
        self.orders = []
        self.ideal_exit = None
        self.window_prices = OrderedDict()


def do_exits(ctx: Context, price: Price):
    current_price = price[1]
    ctx.db.execute(
        """
        SELECT timestamp, interval, timeperiod, threshold
        FROM aroon_exits
        WHERE timestamp = ?""",
        (str(ctx.current_time),),
    )
    exits = ctx.db.fetchall()
    keyed_exits = {}
    for exit in exits:
        exit_id = get_aroon_id(exit)
        keyed_exits[exit_id] = exit
        for entry in ctx.entries:
            entry_time = entry[0]
            sim_id = get_simulation_id(entry, exit)
            sim = ctx.simulations.get(sim_id, None)
            if sim is None:
                open_record = ctx.window_prices.get(entry_time, None)
                open_price = open_record[1]
                sell_price = current_price
                profit = sell_price / open_price
                sim = Simulation(None, Trade(str_to_datetime(entry_time), profit))
                ctx.simulations[sim_id] = sim

                # provide lookup of simulations by buy indicator for use in entries
                entry_id = get_aroon_id(entry)
                bought = ctx.sims_by_entry.get(entry_id, [])
                bought.append(weakref.proxy(sim))
                ctx.sims_by_entry[entry_id] = bought
            elif sim.has_open_position():
                buy_price = ctx.window_prices[sim.buy_time][1]
                sell_price = current_price
                profit = (sell_price * (1 - ctx.fee)) / (buy_price * (1 + ctx.fee))
                sim.close_position(profit)

    return keyed_exits


def do_entries(ctx: Context):
    ctx.db.execute(
        """
        SELECT timestamp, interval, timeperiod, threshold
        FROM aroon_entries
        WHERE timestamp = ?""",
        (str(ctx.current_time),),
    )
    entries = ctx.db.fetchall()
    keyed_entries = {}
    for entry in entries:
        ctx.entries.append(
            Aroon(str_to_datetime(entry[0]), entry[1], entry[2], entry[3])
        )
        entry_id = get_aroon_id(entry)
        keyed_entries[entry_id] = entry
        for sim in ctx.sims_by_entry.get(entry_id, []):
            if not sim.has_open_position:
                sim.open_position(entry["timestamp"])
    return keyed_entries


def send_to_puppy_farm(ctx: Context):
    death_time = ctx.current_time - ctx.delta
    for key, sim in ctx.simulations.items():
        if sim.buy_time == death_time:  # clear sims with old open orders
            del ctx.simulations[key]
        else:
            sim.trim_history(death_time)

    while len(ctx.entries) > 0 and ctx.entries[0][0] <= death_time:
        ctx.entries.pop(0)

    return ctx


def get_total_profit(sim: Simulation):
    total = 1
    for trade in sim.history:
        buy_time, profit = trade
        total *= profit
    return total


def get_best_simulations(
    ctx: Context, margin: number = 0.99, min_profit=1, min_trade_history=3
):
    profits = [
        (key, get_total_profit(sim)) for key, sim in list(ctx.simulations.items())
    ]
    if not len(profits):
        return None

    sorted_profits = sorted(profits, key=lambda x: x[-1], reverse=True)

    best_key, best_profit = sorted_profits[0]

    def qualifier(sim):
        key, profit = sim
        minimum = (best_profit - min_profit) * margin
        return (
            profit > minimum + min_profit
            and len(ctx.simulations[key].history) > min_trade_history
        )

    best = filter(qualifier, sorted_profits)

    return best


def tick(ctx: Context, price: Price, skip_order=False):
    ctx.current_time = pandas.to_datetime(price[0])
    ctx = send_to_puppy_farm(ctx)
    exits = do_exits(ctx, price)
    entries = do_entries(ctx)

    if skip_order:
        return 1

    if len(ctx.orders) == 0 or ctx.orders[-1][1] == Direction.SELL:
        best_simulations = get_best_simulations(ctx)
        if best_simulations is None:
            return 1
        for key, profit in best_simulations:
            [entry_id, exit_id] = key.split(":")
            entry = entries.get(entry_id, None)
            if entry is not None:
                ctx.ideal_exit = exit_id
                break
        ctx.orders.append(Order(ctx.current_time, Direction.BUY))
    else:
        exit = exits.get(ctx.ideal_exit, None)
        if exit is not None and len(ctx.orders) != 0:
            last_order_time = ctx.orders[-1][0]
            buy_price = ctx.window_prices[last_order_time][1]
            sell_price = price[1]
            ctx.orders.append(Order(ctx.current_time, Direction.SELL))
            return (sell_price * (1 - ctx.fee)) / (buy_price * (1 + ctx.fee))
    return 1


def loop(db, window=60, fee=0, **kwargs):
    prices = load_prices(interval=1, **kwargs)
    window_size = 0
    # buys: list((aroonKey, timestamp))
    ctx = Context(
        prices=prices,
        db=db,
        delta=window,
        fee=fee,
    )
    total_profit = 1
    for price in tqdm(prices.itertuples(index=True), total=len(prices.index)):
        if window_size == window:
            ctx.window_prices.popitem(last=False)
        ctx.window_prices[pandas.to_datetime(price[0])] = price
        total_profit *= tick(ctx, price, skip_order=window_size != window)
        gc.collect()
        window_size = window_size + 1 if window_size < window else window

    print("Profit:", total_profit)
    print(ctx.orders)


pair = "XBTUSD"
path = "../data/kraken"
start_date = "2019-05-15"
end_date = "2019-05-15 12:00"
# end_date = "2021-06-16"

conn = sqlite3.connect(database="output/signals/XBTUSD Aroon 2021-04-16T2204.db")
cur = conn.cursor()
loop(
    cur,
    window=30,
    pair=pair,
    path=path,
    startDate=start_date,
    endDate=end_date,
    fee=0.002,
)
conn.close()