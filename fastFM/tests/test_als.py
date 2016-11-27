# Author: Immanuel Bayer
# License: BSD 3 clause

import numpy as np
import scipy.sparse as sp
from sklearn import metrics
from fastFM import als
from fastFM.datasets import make_user_item_regression
from sklearn.metrics import mean_squared_error
from sklearn.utils.testing import assert_almost_equal


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


def get_small_data():
    X = sp.csc_matrix(np.array([[1, 2],
                                [3, 4],
                                [5, 6]]), dtype=np.float64)
    y = np.array([600, 2800, 10000], dtype=np.float64)
    return X, y


def _test_fm_regression_only_w0():
    X, y = get_small_data()

    fm = als.FMRegression(n_iter=0, l2_reg_w=0, l2_reg_V=0, rank=0)
    fm.ignore_w = True
    fm.w0_ = 2
    fm.fit(X, y, warm_start=True)
    assert_almost_equal(fm.w0_, 2, 6)

    fm = als.FMRegression(n_iter=1, l2_reg_w=0, l2_reg_V=0, rank=0)
    fm.ignore_w = True
    fm.w0_ = 2
    fm.fit(X, y, warm_start=True)
    assert_almost_equal(fm.w0_, 4466.6666666666661, 6)


def _test_raise_when_input_is_dense():
    fm = als.FMRegression(n_iter=0, l2_reg_w=0, l2_reg_V=0, rank=0)
    X = np.arange(3, 4, dtype=np.float64)
    y = np.arange(3, dtype=np.float64)
    fm.fit(X, y, warm_start=True)


def test_fm_linear_regression():
    X, y = get_small_data()

    fm = als.FMRegression(n_iter=1, l2_reg_w=1, l2_reg_V=1, rank=0)
    fm.fit(X, y)


def test_fm_regression():
    w0, w, V, y, X = get_test_problem()

    fm = als.FMRegression(n_iter=1000, l2_reg_w=0, l2_reg_V=0, rank=2)
    fm.fit(X, y)
    y_pred = fm.predict(X)
    assert_almost_equal(y_pred, y, 3)
    # check different size
    fm = als.FMRegression(n_iter=1000, l2_reg_w=0, l2_reg_V=0, rank=5)
    X_big = sp.hstack([X, X])
    fm.fit(X_big, y)
    y_pred = fm.predict(X_big[:2, ])


def test_fm_classification():
    w0, w, V, y, X = get_test_problem(task='classification')

    fm = als.FMClassification(n_iter=1000,
                              init_stdev=0.1, l2_reg_w=0, l2_reg_V=0, rank=2)
    fm.fit(X, y)
    y_pred = fm.predict(X)
    print(y_pred)
    assert metrics.accuracy_score(y, y_pred) > 0.95
    # check different size
    fm.fit(X[:2, ], y[:2])


def test_als_warm_start():
    X, y, coef = make_user_item_regression(label_stdev=0)
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    X_train = sp.csc_matrix(X_train)
    X_test = sp.csc_matrix(X_test)

    fm = als.FMRegression(n_iter=10, l2_reg_w=0, l2_reg_V=0, rank=2)
    fm.fit(X_train, y_train)
    y_pred = fm.predict(X_test)
    error_10_iter = mean_squared_error(y_pred, y_test)

    fm = als.FMRegression(n_iter=5, l2_reg_w=0, l2_reg_V=0, rank=2)
    fm.fit(X_train, y_train)
    print(fm.iter_count)
    y_pred = fm.predict(X_test)
    error_5_iter = mean_squared_error(y_pred, y_test)

    fm.fit(sp.csc_matrix(X_train), y_train, n_more_iter=5)
    print(fm.iter_count)
    y_pred = fm.predict(X_test)
    error_5_iter_plus_5 = mean_squared_error(y_pred, y_test)

    print(error_5_iter, error_5_iter_plus_5, error_10_iter)

    assert error_10_iter == error_5_iter_plus_5


def test_warm_start_path():

    X, y, coef = make_user_item_regression(label_stdev=.4)
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    X_train = sp.csc_matrix(X_train)
    X_test = sp.csc_matrix(X_test)
    n_iter = 10

    rank = 4
    seed = 333
    step_size = 1
    l2_reg_w = 0
    l2_reg_V = 0

    fm = als.FMRegression(n_iter=0, l2_reg_w=l2_reg_w,
                          l2_reg_V=l2_reg_V, rank=rank, random_state=seed)
    # initalize coefs
    fm.fit(X_train, y_train)

    rmse_train = []
    rmse_test = []
    for i in range(1, n_iter):
        fm.fit(X_train, y_train, n_more_iter=step_size)
        rmse_train.append(np.sqrt(mean_squared_error(
            fm.predict(X_train), y_train)))
        rmse_test.append(np.sqrt(mean_squared_error(
            fm.predict(X_test), y_test)))

    print('------- restart ----------')
    values = np.arange(1, n_iter)
    rmse_test_re = []
    rmse_train_re = []
    for i in values:
        fm = als.FMRegression(n_iter=i, l2_reg_w=l2_reg_w,
                              l2_reg_V=l2_reg_V, rank=rank, random_state=seed)
        fm.fit(X_train, y_train)
        rmse_test_re.append(np.sqrt(mean_squared_error(
            fm.predict(X_test), y_test)))
        rmse_train_re.append(np.sqrt(mean_squared_error(
            fm.predict(X_train), y_train)))

    assert_almost_equal(rmse_train, rmse_train_re)
    assert_almost_equal(rmse_test, rmse_test_re)


def test_clone():
    from sklearn.base import clone

    a = als.FMRegression()
    b = clone(a)
    assert a.get_params() == b.get_params()

    a = als.FMClassification()
    b = clone(a)
    assert a.get_params() == b.get_params()


if __name__ == '__main__':
    # test_fm_regression_only_w0()
    test_fm_linear_regression()
