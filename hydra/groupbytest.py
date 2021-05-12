import timeit
import pandas as pd

entries = pd.DataFrame(
    index=["2018-08-08", "2018-08-10", "2018-08-12"],
    data={"val": 1},
)

exits = pd.DataFrame(
    index=["2018-08-09", "2018-08-10", "2018-08-11"],
    data={"val": -1},
)


def crossings_nonzero_pos2neg(data):
    pos = data > 0
    return (pos[:-1] & ~pos[1:]).nonzero()[0] + 1


def crossings_nonzero_neg2pos(data):
    pos = data > 0
    return (~pos[:-1] & pos[1:]).nonzero()[0] + 1


def concat():
    return pd.concat([entries, exits]).groupby(level=0)["val"].sum()


def join():
    orders = exits.join(entries, how="outer", lsuffix="ex")
    # return orders.val + orders.valex
    return orders.val.fillna(0) + orders.valex.fillna(0)


# print(concat())
# print("===================")
print(join())
# print("===================")
# print(timeit.timeit("join()", number=10000, globals=globals()))
# print("===================")
# print(timeit.timeit("concat()", number=10000, globals=globals()))
