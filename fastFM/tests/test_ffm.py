# Author: Immanuel Bayer
# License: BSD 3 clause

import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_almost_equal, assert_equal
import ffm


def get_test_problem():
    X = sp.csc_matrix(np.array([[6, 1],
                                [2, 3],
                                [3, 0],
                                [6, 1],
                                [4, 5]]), dtype=np.float64)
    y = np.array([298, 266, 29, 298, 848], dtype=np.float64)
    V = np.array([[6, 0],
                  [5, 8]], dtype=np.float64)
    w = np.array([9, 2], dtype=np.float64)
    w0 = 2
    return w0, w, V, y, X


def test_ffm_predict():
    w0, w, V, y, X = get_test_problem()
    y_pred = ffm.ffm_predict(w0, w, V, X)
    assert_equal(y_pred, y)

if __name__ == '__main__':
    pass
