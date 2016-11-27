# Author: Immanuel Bayer
# License: BSD 3 clause

import numpy as np
import scipy.sparse as sp
from sklearn import metrics
from fastFM import mcmc
from fastFM.datasets import make_user_item_regression
from sklearn.metrics import mean_squared_error
from sklearn.utils.testing import assert_almost_equal, assert_array_equal


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


def test_fm_regression():
    w0, w, V, y, X = get_test_problem()

    fm = mcmc.FMRegression(n_iter=1000, rank=2, init_stdev=0.1)

    y_pred = fm.fit_predict(X, y, X)
    assert metrics.r2_score(y_pred, y) > 0.99


def test_fm_classification():
    w0, w, V, y, X = get_test_problem()
    # transform to labels easier problem then default one
    y_labels = np.ones_like(y)
    y_labels[y < np.mean(y)] = -1

    fm = mcmc.FMClassification(n_iter=1000, init_stdev=0.1, rank=2)
    y_pred = fm.fit_predict_proba(X, y_labels, X)

    fpr, tpr, thresholds = metrics.roc_curve(y_labels, y_pred)
    auc = metrics.auc(fpr, tpr)
    assert auc > 0.95
    y_pred = fm.predict(X[:2, ])


def test_linear_fm_classification():
    w0, w, V, y, X = get_test_problem()
    # transform to labels easier problem then default one
    y_labels = np.ones_like(y)
    y_labels[y < np.mean(y)] = -1

    fm = mcmc.FMClassification(n_iter=1000, init_stdev=0.1, rank=0)
    y_pred = fm.fit_predict_proba(X, y_labels, X)

    fpr, tpr, thresholds = metrics.roc_curve(y_labels, y_pred)
    auc = metrics.auc(fpr, tpr)
    assert auc > 0.95
    y_pred = fm.predict(X[:2, ])


def test_fm_classification_proba():
    w0, w, V, y, X = get_test_problem()
    # transform to labels easier problem then default one
    y_labels = np.ones_like(y)
    y_labels[y < np.mean(y)] = -1

    fm = mcmc.FMClassification(n_iter=1000, init_stdev=0.1, rank=2)
    y_pred_proba = fm.fit_predict_proba(X, y_labels, X)
    y_pred = fm.fit_predict(X, y_labels, X)
    y_pred_proba[y_pred_proba < .5] = -1
    y_pred_proba[y_pred_proba != -1] = 1
    assert_array_equal(y_pred, y_pred_proba)


def test_mcmc_warm_start():
    X, y, coef = make_user_item_regression(label_stdev=0)
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=44)
    X_train = sp.csc_matrix(X_train)
    X_test = sp.csc_matrix(X_test)

    fm = mcmc.FMRegression(n_iter=100, rank=2)
    y_pred = fm.fit_predict(X_train, y_train, X_test)
    error_10_iter = mean_squared_error(y_pred, y_test)

    fm = mcmc.FMRegression(n_iter=50, rank=2)
    y_pred = fm.fit_predict(X_train, y_train, X_test)
    error_5_iter = mean_squared_error(y_pred, y_test)

    y_pred = fm.fit_predict(X_train, y_train, X_test, n_more_iter=50)
    error_5_iter_plus_5 = mean_squared_error(y_pred, y_test)
    print(error_5_iter, error_5_iter_plus_5, error_10_iter)
    print(fm.hyper_param_)
    assert_almost_equal(error_10_iter, error_5_iter_plus_5, decimal=2)


def test_find_init_stdev():
    X, y, coef = make_user_item_regression(label_stdev=.5)
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=44)
    X_train = sp.csc_matrix(X_train)
    X_test = sp.csc_matrix(X_test)

    fm = mcmc.FMRegression(n_iter=10, rank=5)
    best_init_stdev, mse = mcmc.find_init_stdev(fm, X_train, y_train,
                                                stdev_range=[0.2, 0.5, 1.0])
    best_init_stdev_bad, _ = mcmc.find_init_stdev(fm, X_train, y_train,
                                                  stdev_range=[5.])
    print('--' * 30)
    best_init_stdev_vali, mse_vali = mcmc.find_init_stdev(fm,
                                                          X_train, y_train,
                                                          X_test, y_test,
                                                          stdev_range=[
                                                              0.2, 0.5, 1.0])
    assert best_init_stdev < best_init_stdev_bad
    assert best_init_stdev_vali == best_init_stdev
    assert mse_vali > mse


def test_clone():
    from sklearn.base import clone

    a = mcmc.FMRegression()
    b = clone(a)
    assert a.get_params() == b.get_params()

    a = mcmc.FMClassification()
    b = clone(a)
    assert a.get_params() == b.get_params()


if __name__ == "__main__":
    test_linear_fm_classification()
