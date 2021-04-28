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
    buy_time: datetime
    profit: float


class Simulation:
    id: int
    buy_time: datetime  # timestamp | None
    history: Deque[Trade]
    total_profit: float

    def __init__(self, id, trade):
        self.id = id
        self.history = deque()
        self.history.append(trade)
        self.buy_time = None
        self.total_profit = 1

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id

    def has_open_position(self):
        return self.buy_time is not None

    def open_position(self, time: datetime):
        self.buy_time = time

    def close_position(self, profit):
        self.history.append(Trade(self.buy_time, profit))
        self.total_profit *= profit
        self.buy_time = None

    def trim_history(self, time):
        if len(self.history) > 0:
            # gets buy time of oldest trade in history
            if self.history[0][0] == time:
                trade = self.history.popleft()
                # divide by trade profit
                self.total_profit /= trade[1]
        return self


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


def pandasPriceToOpenPrice(price: PandasPrice) -> Price:
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
    window_prices: OrderedDict[datetime, Price]
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
        self.window_prices = OrderedDict()
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
                buy_price = ctx.window_prices.get(entry_time, None)
                sell_price = current_price
                profit = (sell_price * (1 - ctx.fee)) / (buy_price * (1 + ctx.fee))
                sim = Simulation(sim_id, Trade(ctx.current_time, profit))
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
                buy_price = ctx.window_prices[sim.buy_time]
                sell_price = current_price
                profit = (sell_price * (1 - ctx.fee)) / (buy_price * (1 + ctx.fee))
                sim.close_position(profit)

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


def send_to_puppy_farm(ctx: Context):
    death_time = ctx.current_time - ctx.delta
    delete_me = []
    for key, sim in ctx.simulations.items():
        # print(key, sim.buy_time)
        if sim.buy_time == death_time:  # clear sims with old open orders
            delete_me.append(key)
        else:
            sim.trim_history(death_time)
            if len(sim.history) == 0:
                delete_me.append(key)

    for key in delete_me:
        del ctx.simulations[key]

    while len(ctx.entries) > 0 and ctx.entries[0][0] <= death_time:
        ctx.entries.popleft()

    return ctx


def get_best_simulations(
    ctx: Context, margin: float = 0.99, min_profit=1, min_trade_history=3
):
    profits = [
        (key, sim.total_profit) for key, sim in ctx.simulations.items()  # memleak?
    ]
    if not len(profits):
        return []

    sorted_profits = sorted(profits, key=itemgetter(-1), reverse=True)

    best_key, best_profit = sorted_profits[0]

    def qualifier(sim):
        key, profit = sim
        minimum = (best_profit - min_profit) * margin
        return (
            profit > minimum + min_profit
            and len(ctx.simulations[key].history) >= min_trade_history
        )

    best = filter(qualifier, sorted_profits)
    return best


def tick(ctx: Context, price: Price, skip_order=False, use_cached_simulation=True):
    ctx.current_time = price[0]
    ctx = send_to_puppy_farm(ctx)
    exits = do_exits(ctx, price)
    entries = do_entries(ctx)

    if skip_order:
        return 1

    if len(ctx.orders) == 0 or ctx.orders[-1][1] == Direction.SELL:
        if not use_cached_simulation:
            ctx.best_simulations = get_best_simulations(ctx)

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
        exit = exits.get(ctx.ideal_exit, None)
        if exit is not None and len(ctx.orders) != 0:
            buy_price = ctx.buy_price
            sell_price = price[1]
            ctx.orders.append(Order(ctx.current_time, Direction.SELL))
            return (sell_price * (1 - ctx.fee)) / (buy_price * (1 + ctx.fee))
    return 1


def loop(db, window=60, fee=0, **kwargs):
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
    for price in tqdm(prices.itertuples(index=True), total=len(prices.index)):
        price: Price = pandasPriceToOpenPrice(price)
        if tick_count >= window:
            ctx.window_prices.popitem(last=False)
        ctx.window_prices[price[0]] = price[1]
        total_profit *= tick(
            ctx,
            price,
            skip_order=tick_count < window,
            use_cached_simulation=tick_count % round(window / 4) != 0,
        )
        gc.collect()
        tick_count += 1

        print(len(ctx.entries), len(ctx.simulations))
        # items = list(ctx.window_prices.items())
        # print(
        #     tick_count,
        #     len(ctx.window_prices),
        #     items[0],
        # )

    print("Profit:", total_profit)
    print(ctx.orders)


pair = "XBTUSD"
path = "../data/kraken"
start_date = "2019-05-24 00:00"
end_date = "2019-05-27 00:00"
# end_date = "2019-05-28"

conn = sqlite3.connect(database="output/signals/XBTUSD Aroon 2021-04-16T2204.db")
cur = conn.cursor()
loop(
    cur,
    window=120,
    pair=pair,
    path=path,
    startDate=start_date,
    endDate=end_date,
    fee=0.002,
)
conn.close()
