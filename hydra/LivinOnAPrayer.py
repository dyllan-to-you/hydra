import json
import pprint
import sqlite3
import weakref
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
from operator import itemgetter
from typing import Deque, Dict, List, NamedTuple, OrderedDict
import numpy as np
import pandas
from six import Iterator
from tqdm import tqdm

from hydra.PriceLoader import load_prices
from hydra.utils import printd, sanitize_filename, timeme

from numba import njit

pp = pprint.PrettyPrinter(indent=2)


class Direction(Enum):
    BUY = True
    SELL = False


class Order(NamedTuple):
    timestamp: datetime
    order: Direction


class Trade(NamedTuple):
    sim_id: int
    buy_time: datetime
    profit: float


class Simulation:
    id: int
    buy_time: datetime  # timestamp | None
    history_count: int
    total_profit: float

    def __init__(self, id):
        self.id = id

        self.total_profit = 1
        self.buy_time = None
        self.history_count = 0

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id

    def serialize(self):
        return self.__dict__

    def has_open_position(self):
        return self.buy_time is not None

    def open_position(self, time: datetime):
        self.buy_time = time

    def close_position(self, ctx, profit) -> Trade:
        trade = Trade(self.id, self.buy_time, profit)
        self.total_profit *= profit
        self.history_count += 1
        self.buy_time = None
        return trade

    def kill_profit(self, profit):
        self.total_profit /= profit
        self.history_count -= 1


class Aroon(NamedTuple):
    timestamp: datetime
    interval: int
    timeperiod: int
    threshold: int


class Indicator(NamedTuple):
    timestamp: datetime
    id: int


class PandasPrice(NamedTuple):
    timestamp: pandas.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


class Price(NamedTuple):
    timestamp: datetime
    open: float


class TimePeriod(NamedTuple):
    price: Price
    trades: List[Trade]


# @timeme
def pandasPriceToOpenPrice(price: PandasPrice) -> Price:
    """
    Converts pandas price slice to `Price` Object.

    Converts `pandas.Timestamp` to python `datetime`
    """
    timestamp, open, high, low, close, volume = price
    return Price(timestamp.to_pydatetime(), open)


class Context:
    current_time: datetime
    prices: pandas.DataFrame
    db: sqlite3.Cursor
    entries: Deque[Indicator]
    simulations: Dict[str, Simulation]
    sims_by_entry: Dict[str, weakref.WeakSet]
    delta: timedelta
    fee: float
    orders: List[Order]
    ideal_exit: str
    window_periods: OrderedDict[datetime, TimePeriod]
    buy_price: float
    best_simulations: Iterator

    id_base: int

    def __init__(self, prices, db: sqlite3.Cursor, delta, fee, entry_db, exit_db):
        self.prices = prices
        self.db = db
        self.entries = deque()
        self.simulations = dict()
        self.sims_by_entry = dict()
        self.delta = timedelta(minutes=delta)
        self.buy_fee = 1 + fee
        self.sell_fee = 1 - fee
        self.orders = []
        self.ideal_exit = None
        self.window_periods = OrderedDict()
        self.best_simulations = []

        db.execute(
            f"""SELECT MAX(count) FROM (
                SELECT COUNT(*) count FROM {entry_db}
                UNION
                SELECT COUNT(*) count FROM {exit_db});"""
        )
        (base,) = db.fetchone()
        self.id_base = base


# @njit(fastmath=True)
def get_simulation_id(id_base, entry, exit):
    return entry * id_base + exit


# @njit(fastmath=True)
def get_indicator_id(id_base, sim_id):
    entry_id = sim_id // id_base
    exit_id = sim_id % id_base
    return entry_id, exit_id


# @timeme
def get_exits(db, current_time):
    db.execute(
        """
        SELECT id
        FROM aroon_exit
        WHERE timestamp = ?""",
        (str(current_time),),
    )
    return db.fetchall()


# @timeme
def do_exits(ctx: Context, price: Price):
    current_time, current_price = price
    exits = get_exits(ctx.db, current_time)
    keyed_exits = {}
    # entries = np.array(list(ctx.entries))[:, 1]
    for (exit_id,) in exits:
        keyed_exits[exit_id] = current_time
        for _, entry_id in ctx.entries:
            sim_id = get_simulation_id(ctx.id_base, entry_id, exit_id)
            sim = ctx.simulations.get(sim_id, None)
            if sim is None:
                create_sim(ctx, current_price, entry_id, sim_id)
            elif sim.has_open_position():
                add_sim_history(ctx, current_price, sim)

    return keyed_exits


@njit(fastmath=True)
def calculate_profit(buy_fee, sell_fee, buy_price, sell_price):
    return (sell_price * sell_fee) / (buy_price * buy_fee)


def add_sim_history(ctx, current_price, sim):
    buy_price, trades = ctx.window_periods.get(sim.buy_time, (None, None))
    if buy_price is None:
        raise Exception(f"ERROR: Could not find window_price for {sim.buy_time=}")
    sell_price = current_price

    profit = calculate_profit(ctx.buy_fee, ctx.sell_fee, buy_price, sell_price)
    trade = sim.close_position(ctx, profit)
    trades.append(trade)


def create_sim(ctx, current_price, entry_id, sim_id):
    sim = Simulation(sim_id)
    sim.open_position(ctx.current_time)

    add_sim_history(ctx, current_price, sim)

    ctx.simulations[sim_id] = sim
    # provide lookup of simulations by buy indicator for use in entries
    bought: weakref.WeakSet = ctx.sims_by_entry.get(entry_id, None)
    if bought is None:
        bought = weakref.WeakSet()
        bought.add(sim)
        ctx.sims_by_entry[entry_id] = bought
    else:
        bought.add(sim)


# @timeme
def get_entries(db, current_time):
    db.execute(
        """
        SELECT id
        FROM aroon_entry
        WHERE timestamp = ?""",
        (str(current_time),),
    )
    return db.fetchall()


# @timeme
def do_entries(ctx: Context):
    entries = get_entries(ctx.db, ctx.current_time)
    keyed_entries = {}
    for (entry_id,) in entries:
        ctx.entries.append(Indicator(ctx.current_time, entry_id))
        keyed_entries[entry_id] = ctx.current_time
        for sim in ctx.sims_by_entry.get(entry_id, []):
            sim: Simulation = sim
            if not sim.has_open_position():
                sim.open_position(ctx.current_time)
    return keyed_entries


# @timeme
def get_best_simulations(
    ctx: Context, margin: float = 0.99, min_profit=1.03, min_trade_history=3
):

    profits = [
        (key, sim.total_profit, sim.history_count)
        for key, sim in ctx.simulations.items()
        # decreases the effectiveness of caching sorted profits
        if sim.history_count >= min_trade_history and sim.total_profit >= min_profit
    ]
    if not len(profits):
        return []

    sorted_profits = sorted(profits, key=itemgetter(1), reverse=True)
    best_profit = sorted_profits[0][1]

    best_profit = max(profits, key=itemgetter(1))[1]

    def qualifier(sim):
        key, profit, history = sim
        minimum = (best_profit - min_profit) * margin
        return profit > minimum + min_profit

    best = filter(qualifier, profits)
    best_sorted = sorted(best, key=itemgetter(2), reverse=True)
    return list(best_sorted)

    sorted_profits = sorted(profits, key=itemgetter(1, -1), reverse=True)

    best_profit = sorted_profits[0][1]

    def qualifier(sim):
        key, profit, history = sim
        minimum = (best_profit - min_profit) * margin
        return profit > minimum + min_profit

    best = filter(qualifier, sorted_profits)
    return list(best)


# @timeme
def tick(
    ctx: Context, price: PandasPrice, window_is_full=True, use_cached_simulation=True
):
    price: Price = pandasPriceToOpenPrice(price)
    current_time, open_price = price
    if window_is_full:
        death_date, period = ctx.window_periods.popitem(last=False)
        send_to_puppy_farm(ctx, death_date, period)
    ctx.window_periods[current_time] = (open_price, [])

    ctx.current_time = current_time
    exits = do_exits(ctx, price)
    entries = do_entries(ctx)
    cost = (len(ctx.entries) * len(exits)) + len(entries)
    # printd(
    #     f"{ctx.current_time} {len(ctx.simulations)=} {len(ctx.entries)=} {len(entries)=} {len(exits)=} {cost=}"
    # )
    if not window_is_full:
        return 1

    if len(ctx.orders) == 0 or ctx.orders[-1][1] == Direction.SELL:
        if not use_cached_simulation:
            ctx.best_simulations = get_best_simulations(ctx)
        # printd("Best (Seeking Buy)", len(ctx.best_simulations))
        for key, profit, history_len in ctx.best_simulations:
            entry_id, exit_id = get_indicator_id(ctx.id_base, key)
            entry = entries.get(entry_id, None)
            if entry is not None:
                ctx.ideal_exit = exit_id
                ctx.orders.append(
                    Order(
                        ctx.current_time,
                        Direction.BUY,
                    )
                )
                ctx.buy_price = open_price
                break
    else:
        # printd("Best (Seeking Sell)", len(ctx.best_simulations))
        exit = exits.get(ctx.ideal_exit, None)
        if exit is not None and len(ctx.orders) != 0:
            buy_price = ctx.buy_price
            sell_price = open_price
            ctx.orders.append(Order(ctx.current_time, Direction.SELL))
            return (sell_price * ctx.sell_fee) / (buy_price * ctx.buy_fee)
    return 1


# @timeme
def send_to_puppy_farm(ctx: Context, death_date: datetime, period: TimePeriod):
    for sim_id, buy_time, profit in period[1]:
        # Update simulation profit
        sim = ctx.simulations[sim_id]
        sim.kill_profit(profit)
        if sim.history_count == 0:
            del ctx.simulations[sim_id]

    while len(ctx.entries) > 0 and ctx.entries[0][0] <= death_date:
        ctx.entries.popleft()

    return ctx


# @timeme
def loop(db, window=60, fee=0, snapshot=False, **kwargs):
    prices = load_prices(interval=1, **kwargs)
    tick_count = 0
    # buys: list((aroonKey, timestamp))
    ctx = Context(
        prices=prices,
        delta=window,
        fee=fee,
        db=db,
        entry_db="aroon_entry_ids",
        exit_db="aroon_exit_ids",
    )

    total_profit = 1
    for price in tqdm(
        prices.itertuples(index=True),
        total=len(prices.index),
        leave=True,
        unit="tick",
    ):
        total_profit *= tick(
            ctx,
            price,
            window_is_full=tick_count >= window,
            use_cached_simulation=tick_count % 15 != 0,
        )

        # timeme(gc.collect)()
        tick_count += 1

    if snapshot:
        with open(
            f"{sanitize_filename(kwargs['startDate'])}.{len(ctx.simulations)} simulations.txt",
            "w",
        ) as outfile:
            # TODO: Ensure whatever i'm trying to dump supports it
            json.dump(
                ctx.simulations,
                outfile,
                indent=2,
                default=lambda o: o.serialize()
                if hasattr(o, "serialize")
                else str(o)
                if isinstance(o, datetime)
                else o.__dict__,
            )

    printd("Profit:", total_profit)
    printd(json.dumps(ctx.orders, default=str))


pair = "XBTUSD"
path = "../data/kraken"
start_date = "2019-05-20 12:00"
end_date = "2019-05-21 00:00"

conn = sqlite3.connect(database="output/signals/XBTUSD Aroon 2021-04-16T2204.db")
cur = conn.cursor()

loop(
    cur,
    window=300,
    pair=pair,
    path=path,
    startDate=start_date,
    endDate=end_date,
    fee=0.0016,
    # snapshot=True,
)
conn.close()
