import numpy as np


def test_min():
    index_min = np.argmin([5, 1, 2, 3, 2, 1, 2])
    print(index_min)
