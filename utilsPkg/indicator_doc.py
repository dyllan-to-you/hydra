import numpy as np
import tulipy as ti


def print_info(indicator):
    print("Type:", indicator.type)
    print("Full Name:", indicator.full_name)
    print("Inputs:", indicator.inputs)
    print("Options:", indicator.options)
    print("Outputs:", indicator.outputs)


print_info(ti.psar)

high = np.array(
    [
        81.59,
        81.06,
        82.87,
        83,
        83.61,
        83.15,
        82.84,
        83.99,
        84.55,
        84.36,
        85.53,
        86.54,
        86.89,
        87.77,
        87.29,
    ]
)

low = high * 0.95

# 15 inputs = 14 outputs
print(
    ti.psar(high, low, acceleration_factor_step=0.025, acceleration_factor_maximum=0.25)
)
