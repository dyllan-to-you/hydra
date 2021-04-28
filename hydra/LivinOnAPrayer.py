import json
from hydra.utils import sanitize_filename
from operator import itemgetter
from collections import deque
from enum import Enum
import gc
from typing import Deque, Dict, List, NamedTuple, OrderedDict
import weakref
import pandas
from six import Iterator
from hydra.SuperSim import load_prices
import sqlite3
from datetime import datetime, timedelta
from tqdm import tqdm
import pprint

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
        self.fee = fee
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

    def get_simulation_id(self, entry, exit):
        return entry * self.id_base + exit

    def get_indicator_id(self, sim_id):
        entry_id = sim_id // self.id_base
        exit_id = sim_id % self.id_base
        return entry_id, exit_id


def do_exits(ctx: Context, price: Price):
    current_price = price[1]
    ctx.db.execute(
        """
        SELECT id
        FROM aroon_exit
        WHERE timestamp = ?""",
        (str(ctx.current_time),),
    )
    exits = ctx.db.fetchall()
    keyed_exits = {}
    for (exit_id,) in exits:
        keyed_exits[exit_id] = ctx.current_time
        for entry_time, entry_id in ctx.entries:
            sim_id = ctx.get_simulation_id(entry_id, exit_id)
            sim = ctx.simulations.get(sim_id, None)
            if sim is None:
                buy_price, trades = ctx.window_periods.get(entry_time, (None, None))
                if buy_price is None:
                    raise Exception(
                        f"ERROR could not find window_price for {entry_time=}"
                    )
                sell_price = current_price
                profit = (sell_price * (1 - ctx.fee)) / (buy_price * (1 + ctx.fee))
                sim = Simulation(sim_id)
                sim.open_position(ctx.current_time)
                trade = sim.close_position(ctx, profit)
                trades.append(trade)
                ctx.simulations[sim_id] = sim

                # provide lookup of simulations by buy indicator for use in entries
                bought: weakref.WeakSet = ctx.sims_by_entry.get(entry_id, None)
                if bought is None:
                    bought = weakref.WeakSet()
                    bought.add(sim)
                    ctx.sims_by_entry[entry_id] = bought
                else:
                    bought.add(sim)
            elif sim.has_open_position():
                buy_price, trades = ctx.window_periods.get(sim.buy_time, (None, None))
                if buy_price is None:
                    raise Exception(
                        f"ERROR: Could not find window_price for {sim.buy_time=}"
                    )
                sell_price = current_price
                profit = (sell_price * (1 - ctx.fee)) / (buy_price * (1 + ctx.fee))
                trade = sim.close_position(ctx, profit)
                trades.append(trade)

    return keyed_exits


def do_entries(ctx: Context):
    ctx.db.execute(
        """
        SELECT id
        FROM aroon_entry
        WHERE timestamp = ?""",
        (str(ctx.current_time),),
    )
    entries = ctx.db.fetchall()
    keyed_entries = {}
    for (entry_id,) in entries:
        ctx.entries.append(Indicator(ctx.current_time, entry_id))
        keyed_entries[entry_id] = ctx.current_time
        for sim in ctx.sims_by_entry.get(entry_id, []):
            sim: Simulation = sim
            if not sim.has_open_position():
                sim.open_position(ctx.current_time)
    return keyed_entries


def get_best_simulations(
    ctx: Context, margin: float = 0.99, min_profit=1, min_trade_history=3
):
    profits = [
        (key, sim.total_profit)
        for key, sim in ctx.simulations.items()
        # decreases the effectiveness of caching sorted profits
        if sim.history_count >= min_trade_history
    ]
    if not len(profits):
        return []

    sorted_profits = sorted(profits, key=itemgetter(-1), reverse=True)

    best_key, best_profit = sorted_profits[0]

    def qualifier(sim):
        key, profit = sim
        minimum = (best_profit - min_profit) * margin
        return profit > minimum + min_profit

    best = filter(qualifier, sorted_profits)
    return list(best)


def tick(ctx: Context, price: Price, skip_order=False, use_cached_simulation=True):
    ctx.current_time = price[0]
    exits = do_exits(ctx, price)
    entries = do_entries(ctx)
    cost = len(ctx.entries) * len(exits)
    print(f"{ctx.current_time} {len(entries)=} {len(exits)=} {cost=}")
    if skip_order:
        return 1

    if len(ctx.orders) == 0 or ctx.orders[-1][1] == Direction.SELL:
        if not use_cached_simulation:
            ctx.best_simulations = get_best_simulations(ctx)
        print("Best(Waiting For Buy)", len(ctx.best_simulations))
        for key, profit in ctx.best_simulations:
            entry_id, exit_id = ctx.get_indicator_id(key)
            entry = entries.get(entry_id, None)
            if entry is not None:
                ctx.ideal_exit = exit_id
                ctx.orders.append(
                    Order(
                        ctx.current_time,
                        Direction.BUY,
                    )
                )
                ctx.buy_price = price[1]
                break
    else:
        print("Best(Waiting For Sell)", len(ctx.best_simulations))
        exit = exits.get(ctx.ideal_exit, None)
        if exit is not None and len(ctx.orders) != 0:
            buy_price = ctx.buy_price
            sell_price = price[1]
            ctx.orders.append(Order(ctx.current_time, Direction.SELL))
            return (sell_price * (1 - ctx.fee)) / (buy_price * (1 + ctx.fee))
    return 1


def send_to_puppy_farm(ctx: Context, death_date: datetime, period: TimePeriod):
    deleted = 0
    print(f"[{death_date}] Trimming profits from {len(period[1])} simulations.")
    for trade in period[1]:
        # Update simulation profit
        sim = ctx.simulations[trade[0]]
        sim.kill_profit(trade[2])
        if sim.history_count == 0:
            del ctx.simulations[trade[0]]
            deleted += 1

    print("Murdered", deleted, "simulations")

    while len(ctx.entries) > 0 and ctx.entries[0][0] <= death_date:
        ctx.entries.popleft()

    return ctx


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
        prices.itertuples(index=True), total=len(prices.index), smoothing=1, leave=True
    ):
        price: Price = pandasPriceToOpenPrice(price)
        if tick_count >= window:
            death_date, period = ctx.window_periods.popitem(last=False)
            send_to_puppy_farm(ctx, death_date, period)
        ctx.window_periods[price[0]] = (price[1], [])

        total_profit *= tick(
            ctx,
            price,
            skip_order=tick_count < window,
            use_cached_simulation=tick_count % round(window / 4) != 0,
        )

        gc.collect()
        tick_count += 1
        print("Total", len(ctx.entries), len(ctx.simulations))

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

    print("Profit:", total_profit)
    print(ctx.orders)


pair = "XBTUSD"
path = "../data/kraken"
start_date = "2019-05-24 00:00"
# end_date = "2019-05-24 02:00"
end_date = "2019-05-25 00:00"

conn = sqlite3.connect(database="output/signals/XBTUSD Aroon 2021-04-16T2204.db")
cur = conn.cursor()

loop(
    cur,
    window=180,
    pair=pair,
    path=path,
    startDate=start_date,
    endDate=end_date,
    fee=0.002,
    # snapshot=True,
)
conn.close()
