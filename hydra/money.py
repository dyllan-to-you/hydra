from numba import njit


@njit(cache=True, fastmath=True)
def get_decay_profit(original, rate, profit, minutes_elapsed=1):
    return ((original * profit - 1) * (rate ** minutes_elapsed)) + 1


@njit(cache=True, fastmath=True)
def get_decay(original, rate, minutes_elapsed=1):
    return original * 0.75


# @njit(cache=True, fastmath=True)
# def get_decay(original, rate, minutes_elapsed=1):
#     return original * (rate ** minutes_elapsed)


@njit(cache=True, fastmath=True)
def calculate_profit(buy_price, sell_price, buy_fee=1, sell_fee=1) -> float:
    return (sell_price * sell_fee) / (buy_price * buy_fee)
