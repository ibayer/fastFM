# Author: Immanuel Bayer
# License: BSD 3 clause

import numpy as np
from fastFM.utils import kendall_tau


def test_ffm_vector_kendall_tau():
    order = np.array([1, 2, 3, 4, 5])
    order_wrong = np.array([5, 3, 4, 2, 1])
    order_inv = np.array([5, 4, 3, 2, 1])

    assert kendall_tau(order, order) == 1
    assert kendall_tau(order, order_inv) == -1
    assert kendall_tau(order, order_wrong) != -1


if __name__ == '__main__':
    test_ffm_vector_kendall_tau()
