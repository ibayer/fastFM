# Author: Immanuel Bayer
# License: BSD 3 clause
import json

import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_equal
from sklearn.metrics import mean_squared_error

import ffm
import ffm2

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

def test_ffm2_predict():
    w0, w, V, y, X = get_test_problem()
    y_pred = ffm2.ffm_predict(w0, w, V, X)
    assert_equal(y_pred, y)

def test_ffm2_predict_w0():
    w0, w, V, y, X = get_test_problem()
    w[:] = 0
    V[:, :] = 0
    y_pred = ffm2.ffm_predict(w0, w, V, X)
    assert_equal(y_pred, w0)

def test_ffm2_fit():
    w0, w, V, y, X = get_test_problem()
    w0 = 0
    w[:] = 0
    np.random.seed(123)
    V = np.random.normal(loc=0.0, scale=1.0,
                         size=(2, 2))

    w0_init = w0
    w_init = np.copy(w)
    V_init = np.copy(V)
    rank = 2

    y_pred = ffm2.ffm_predict(w0, w, V, X)
    msqr_before = mean_squared_error(y, y_pred)

    jsn = json.dumps({'solver': 'cd',
                      'loss': 'squared',
                      'iter': 5,
                      'l2_reg_w': 0.01,
                      'l2_reg_V': 0.02}).encode()

    w0, w, V = ffm2.ffm_als_fit(w0, w, V, X, y, rank, jsn)

    y_pred = ffm2.ffm_predict(w0, w, V, X)
    msqr_after = mean_squared_error(y, y_pred)

    assert w0 != 0
    assert(msqr_before > msqr_after)
