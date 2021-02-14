from hydra.indicators import Aroon, AroonTulip
import pandas as pd

prices = [
    {"Low": 1.0, "High": 2.0},
    {"Low": 2.0, "High": 4.0},
    {"Low": 10.0, "High": 20.0},
    {"Low": 3.0, "High": 6.0},
    {"Low": 9.0, "High": 18.0},
    {"Low": 3.0, "High": 6.0},
    {"Low": 8.0, "High": 16.0},
    {"Low": 6.0, "High": 12.0},
    {"Low": 4.0, "High": 8.0},
    {"Low": 5.0, "High": 10.0},
    {"Low": 5.0, "High": 10.0},
    {"Low": 6.0, "High": 12.0},
    {"Low": 3.0, "High": 6.0},
    {"Low": 7.0, "High": 14.0},
    {"Low": 4.0, "High": 8.0},
]

A = Aroon.Indicator(3)
T = AroonTulip.Indicator(3)

history = []
aroon = []
tulip = []
for price in prices:
    # print(price)
    a = A.calculate(price, history)
    a["price"] = price
    t = T.calculate(price, history)
    t["price"] = price
    aroon.append(a)
    tulip.append(t)
    history.append(price)

cols = [
    "price.Low",
    "price.High",
    "aroonTA(3).up",
    "aroonTA(3).down",
    "aroonTA(3).oscillator",
    "aroonTA(3).peak",
    "aroonTA(3).valley",
]

print(pd.json_normalize(aroon, sep="."))
print(pd.json_normalize(tulip, sep="."))
# print(history, aroon, tulip)