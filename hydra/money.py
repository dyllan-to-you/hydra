from numba import njit


@njit(cache=True)
def get_decay(original, rate, profit, minutes_elapsed=1):
    return ((original * profit - 1) * (rate ** minutes_elapsed)) + 1


@njit(cache=True)
def calculate_profit(buy_price, sell_price, buy_fee=1, sell_fee=1):
    return (sell_price * sell_fee) / (buy_price * buy_fee)
