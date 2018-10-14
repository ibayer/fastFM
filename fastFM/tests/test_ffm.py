# Author: Immanuel Bayer
# License: BSD 3 clause

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
    w0 = np.array([2], dtype=np.float64)
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
    assert_equal(y_pred[0], w0)

def test_ffm2_fit_als():
    w0, w, V, y, X = get_test_problem()
    w0[:] = 0
    w[:] = 0
    np.random.seed(123)
    V = np.random.normal(loc=0.0, scale=1.0,
                         size=(2, 2))
    rank = 2

    y_pred = ffm2.ffm_predict(w0, w, V, X)
    msqr_before = mean_squared_error(y, y_pred)

    settings = {'solver': 'cd',
                'loss': 'squared',
                'iter': 500,
                'l2_reg_w': 0.01,
                'l2_reg_V': 0.02}

    ffm2.ffm_fit(w0, w, V, X, y, rank, settings)

    y_pred = ffm2.ffm_predict(w0, w, V, X)
    msqr_after = mean_squared_error(y, y_pred)

    assert w0 != 0
    assert(msqr_before > msqr_after)

def test_ffm2_fit_sgd():
    w0, w, V, y, X = get_test_problem()
    w0[:] = 0
    w[:] = 0
    np.random.seed(123)
    V = np.random.normal(loc=0.0, scale=0.01,
                         size=(2, 2))

    rank = 2

    y_pred = ffm2.ffm_predict(w0, w, V, X)
    msqr_before = mean_squared_error(y, y_pred)

    settings = {'solver': 'sgd',
                'loss': 'squared',
                'step_size': 0.0001,
                'n_epoch': 5,
                'l2_reg_w': 0.01,
                'l2_reg_V': 0.02}

    w0, w, V = ffm2.ffm_fit(w0, w, V, sp.csr_matrix(X), y, rank, settings)

    y_pred = ffm2.ffm_predict(w0, w, V, X)
    msqr_after = mean_squared_error(y, y_pred)

    assert w0 != 0
    assert(msqr_before > msqr_after)


if __name__ == "__main__":
    # test_ffm2_fit_sgd()
    test_ffm2_fit_als()
