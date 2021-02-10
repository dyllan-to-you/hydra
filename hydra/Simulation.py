from hydra.strategies import Decision, DecisionEvent
import os
import pandas as pd
from csv import DictReader
from typing import cast
from unittest.mock import patch
import dateutil.parser as parser
from hydra import Hydra
from hydra.types import Price
from hydra.strategies.AroonStrategy import AroonStrategy
from tqdm import tqdm
import asyncio


def pick_price_avg(row):
    return (row["Open"] + row["Close"]) / 2


class Simulate:
    def __init__(self, hydras):
        self.hydras = hydras

    def run(self, pick_price=pick_price_avg):
        res = []
        with open(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "../data/Bitfinex_BTCUSD_1h.test.csv",
            )
        ) as file:
            prices = DictReader(file)
            for idx, row in tqdm(enumerate(prices), desc="Processing Time"):
                # if idx > 8760:
                #     break

                price = Price(
                    parser.parse(cast(str, row.get("date"))),
                    float(cast(str, row.get("open"))),
                    float(cast(str, row.get("high"))),
                    float(cast(str, row.get("low"))),
                    float(cast(str, row.get("close"))),
                    float(cast(str, row.get("Volume BTC"))),
                    float(cast(str, row.get("Volume USD"))),
                )
                for hydra in self.hydras:
                    hydra.feed(price)
            for hydra in self.hydras:
                decision_history = hydra.strategy.decision_history_df
                priced_decisions = hydra.price_history_df.join(
                    decision_history, how="right"
                )

                buy_decision = None
                cash = 1
                history = []
                for index, row in priced_decisions.iterrows():
                    if buy_decision is None:
                        cash *= 1 - (hydra.fee / 100)
                        buy_decision = row
                        continue

                    buy = pick_price(buy_decision)
                    sell = pick_price(row)

                    pl = sell / buy
                    cash *= pl
                    cash *= 1 - (hydra.fee / 100)

                    history.append((pl, cash))

                    buy_decision = None
                    pass

                res.append(
                    {
                        "name": hydra.name,
                        "strategy": hydra.strategy.name,
                        "Total": history[-1][1],
                        "Transactions": len(history),
                    }
                )
            return res


# aroons = [
#     (simulate(Hydra(AroonStrategy((i + 1) * 5))), i + 1 * 5)
#     for i, x in enumerate([0] * 9)
# ]
# for (sim, period) in aroons:
#     print(period, sim[-1])
res = []
for i in tqdm(range(1, 10), "Aroons"):
    period = i * 5
    hydras = [
        Hydra(AroonStrategy(period), name="free trade"),
        Hydra(AroonStrategy(period), name="kraken (taker)", fee=0.26),
        Hydra(AroonStrategy(period), name="kraken (maker)", fee=0.16),
        Hydra(AroonStrategy(period), name="binance", fee=0.1),
        # [
        #     "binance(pro)",
        #     period,
        #     simulate(Hydra(AroonStrategy(period)), fee=0.075),
        # ],
        # [
        #     "binance(pro + ref)",
        #     period,
        #     simulate(Hydra(AroonStrategy(period)), fee=0.075 * 0.8),
        # ],
    ]
    sim = Simulate(hydras)
    res.extend(sim.run())

with pd.option_context(
    "display.max_rows", None, "display.max_columns", None
):  # more options can be specified also
    df = pd.DataFrame(res)
    print(df)
