import numpy as np
import scipy.sparse as sp
from sklearn.datasets import make_regression
from sklearn import linear_model
from sklearn import metrics
from numpy.testing import assert_almost_equal, assert_equal
from fastFM.fastfm import FactorizationMachine
import scipy.sparse as sp
from fastFM import utils


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

def test_fm_als_regression():
    w0, w, V, y, X = get_test_problem()

    fm = FactorizationMachine(task='regression', solver='als', max_iter=1000,
            lambda_w=0, lambda_V=0, rank_pair=2)
    fm.fit(X, y)
    y_pred = fm.predict(X)
    assert_almost_equal(y_pred, y, 3)
    # check different size
    fm = FactorizationMachine(task='regression', solver='als', max_iter=1000,
            lambda_w=0, lambda_V=0, rank_pair=5)
    X_big = sp.hstack([X,X]).tocsc()
    fm.fit(X_big, y)
    y_pred = fm.predict(X_big[:2,])


def test_fm_als_classification():
    w0, w, V, y, X = get_test_problem(task='classification')

    fm = FactorizationMachine(task='classification', solver='als', max_iter=1000,
            init_stdev=0.1, lambda_w=0, lambda_V=0, rank_pair=2)
    fm.fit(X, y)
    y_pred = fm.predict(X)
    print y_pred
    assert metrics.accuracy_score(y, y_pred) > 0.95
    # check different size
    fm.fit(X[:2,], y[:2])
    y_pred = fm.predict(X[:2,])

def test_fm_sgd_regression():
    w0, w, V, y, X = get_test_problem()
    X_test = X.copy()
    X_train = sp.csc_matrix(X.T)

    fm = FactorizationMachine(task='regression', solver='sgd', max_iter=10000,
            init_stdev=0.01, lambda_w=0.5, lambda_V=50.5, rank_pair=2,
            step_size=0.0001)

    fm.fit(X_train, y)
    y_pred = fm.predict(X_test)
    assert metrics.r2_score(y_pred, y) > 0.99


def test_fm_sgd_classification():
    w0, w, V, y, X = get_test_problem(task='classification')
    X_test = X.copy()
    X_train = sp.csc_matrix(X.T)

    fm = FactorizationMachine(task='classification', solver='sgd', max_iter=1000,
            init_stdev=0.1, lambda_w=0, lambda_V=0, rank_pair=2)
    fm.fit(X_train, y)
    y_pred = fm.predict(X_test)
    print y_pred
    assert metrics.accuracy_score(y, y_pred) > 0.95

def test_fm_mcmc_regression():
    w0, w, V, y, X = get_test_problem()

    fm = FactorizationMachine(task='regression', solver='mcmc',
            max_iter=1000, rank_pair=2, init_stdev=0.1,
            lambda_w=1, lambda_V=1)

    y_pred = fm.fit_predict(X, y, X)
    assert metrics.r2_score(y_pred, y) > 0.99

def test_fm_mcmc_classification():
    w0, w, V, y, X = get_test_problem()
    # transform to labels easier problem then default one
    y_labels = np.ones_like(y)
    y_labels[y < np.mean(y)] = -1

    fm = FactorizationMachine(task='classification', solver='mcmc', max_iter=1000,
            init_stdev=0.1, lambda_w=0, lambda_V=0, rank_pair=2)
    y_pred = fm.fit_predict(X, y_labels, X)

    fpr, tpr, thresholds = metrics.roc_curve(y_labels, y_pred)
    auc = metrics.auc(fpr, tpr)
    assert auc > 0.95


def test_fm_sgr_ranking():
    w0, w, V, y, X = get_test_problem()
    X_test = X.copy()
    X_train = sp.csc_matrix(X.T)

    import itertools
    pairs = [p for p in itertools.combinations(range(len(y)), 2)]
    compares = np.zeros((len(pairs), 2), dtype=np.float64)

    for i, p in enumerate(pairs):
        if y[p[0]] > y[p[1]]:
            compares[i, 0] = p[0]
            compares[i, 1] = p[1]
        else:
            compares[i, 0] = p[1]
            compares[i, 1] = p[0]

    print compares
    fm = FactorizationMachine(task='ranking', solver='sgd', max_iter=2000,
            init_stdev=0.01, lambda_w=.5, lambda_V=.5, rank_pair=2,
            step_size=.002, random_state=11)
    fm.fit(X_train, compares)
    y_pred = fm.predict(X_test)
    print y
    print y_pred
    print np.argsort(y)
    assert utils.kendall_tau(np.argsort(y), y_pred) == 1


if __name__ == '__main__':
    #test_fm_mcmc_reg()
    #test_fm_sgd_regression()
    #test_fm_sgd_classification()
    test_fm_sgr_ranking()
