# Author: Immanuel Bayer
# License: BSD 3 clause

import numpy as np
import scipy.sparse as sp
from nose.tools import assert_true
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

def get_another_test_problem():
    X = sp.csc_matrix(np.array([[6, 1],
                                [2, 3],
                                [3, 0],
                                [6, 1],
                                [4, 5]]), dtype=np.float64)
    y = np.array([298, 266, 29, 298, 848], dtype=np.float64)
    V = np.random.normal(loc=0.0, scale=1.0, size=(2, 2))
    w = np.array([0, 0], dtype=np.float64)
    w0 = 0
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
    w0, w, V, y, X = get_another_test_problem()
    w0_init = w0
    w_init = np.copy(w)
    V_init = np.copy(V)
    rank = 2

    y_pred = ffm2.ffm_predict(w0, w, V, X)
    msqr_before = mean_squared_error(y, y_pred)

    w0, w, V = ffm2.ffm_als_fit(w0, w, V, X, y, rank)

    y_pred = ffm2.ffm_predict(w0, w, V, X)
    msqr_after = mean_squared_error(y, y_pred)

    assert_true(w0 != w0_init)
    assert_true(np.any(np.not_equal(w, w_init)))
    assert_true(np.any(np.not_equal(V, V_init)))
    assert_true(msqr_before > msqr_after)
