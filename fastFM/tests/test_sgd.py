# Author: Immanuel Bayer
# License: BSD 3 clause

import numpy as np
import scipy.sparse as sp
from sklearn import metrics
from sklearn.datasets import make_regression
from sklearn.utils.testing import assert_almost_equal
from fastFM import sgd
from fastFM import als


def get_test_problem(task='regression'):
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
    if task == 'classification':
        y_labels = np.ones_like(y)
        y_labels[y < np.median(y)] = -1
        y = y_labels
    return w0, w, V, y, X


def test_sgd_regression_small_example():
    w0, w, V, y, X = get_test_problem()
    X_test = X.copy()
    X_train = sp.csc_matrix(X)

    fm = sgd.FMRegression(n_iter=10000,
                          init_stdev=0.01, l2_reg_w=0.5, l2_reg_V=50.5, rank=2,
                          step_size=0.0001)

    fm.fit(X_train, y)
    y_pred = fm.predict(X_test)
    assert metrics.r2_score(y_pred, y) > 0.99


def test_first_order_sgd_vs_als_regression():
    X, y = make_regression(n_samples=100, n_features=50, random_state=123)
    X = sp.csc_matrix(X)

    fm_sgd = sgd.FMRegression(n_iter=900, init_stdev=0.01, l2_reg_w=0.0,
                              l2_reg_V=50.5, rank=0, step_size=0.01)
    fm_als = als.FMRegression(n_iter=10, l2_reg_w=0, l2_reg_V=0, rank=0)

    y_pred_sgd = fm_sgd.fit(X, y).predict(X)
    y_pred_als = fm_als.fit(X, y).predict(X)

    score_als = metrics.r2_score(y_pred_als, y)
    score_sgd = metrics.r2_score(y_pred_sgd, y)

    assert_almost_equal(score_als, score_sgd, decimal=2)


def test_second_order_sgd_vs_als_regression():
    X, y = make_regression(n_samples=100, n_features=50, random_state=123)
    X = sp.csc_matrix(X)

    fm_sgd = sgd.FMRegression(n_iter=50000, init_stdev=0.00, l2_reg_w=0.0,
                              l2_reg_V=50.5, rank=2, step_size=0.0002)
    fm_als = als.FMRegression(n_iter=10, l2_reg_w=0, l2_reg_V=0, rank=2)

    y_pred_als = fm_als.fit(X, y).predict(X)
    y_pred_sgd = fm_sgd.fit(X, y).predict(X)

    score_als = metrics.r2_score(y_pred_als, y)
    score_sgd = metrics.r2_score(y_pred_sgd, y)

    assert_almost_equal(score_sgd, score_als, decimal=2)


def test_sgd_classification_small_example():
    w0, w, V, y, X = get_test_problem(task='classification')
    X_test = X.copy()
    X_train = sp.csc_matrix(X)

    fm = sgd.FMClassification(n_iter=1000,
                              init_stdev=0.1, l2_reg_w=0, l2_reg_V=0, rank=2,
                              step_size=0.1)
    fm.fit(X_train, y)
    y_pred = fm.predict(X_test)
    print(y_pred)
    assert metrics.accuracy_score(y, y_pred) > 0.95


def test_clone():
    from sklearn.base import clone

    a = sgd.FMRegression()
    b = clone(a)
    assert a.get_params() == b.get_params()

    a = sgd.FMClassification()
    b = clone(a)
    assert a.get_params() == b.get_params()


if __name__ == '__main__':
    test_sgd_regression_small_example()
    test_first_order_sgd_vs_als_regression()
    test_second_order_sgd_vs_als_regression()
