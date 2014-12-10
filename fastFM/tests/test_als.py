import numpy as np
import scipy.sparse as sp
from sklearn import metrics
from fastFM import als
from numpy.testing import assert_almost_equal


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

    fm = als.FMRegression(n_iter=1000, l2_reg_w=0, l2_reg_V=0, rank=2)
    fm.fit(X, y)
    y_pred = fm.predict(X)
    assert_almost_equal(y_pred, y, 3)
    # check different size
    fm = als.FMRegression(n_iter=1000, l2_reg_w=0, l2_reg_V=0, rank=5)
    X_big = sp.hstack([X,X]).tocsc()
    fm.fit(X_big, y)
    y_pred = fm.predict(X_big[:2,])


def test_fm_classification():
    w0, w, V, y, X = get_test_problem(task='classification')

    fm = als.FMClassification(n_iter=1000,
            init_stdev=0.1, l2_reg_w=0, l2_reg_V=0, rank=2)
    fm.fit(X, y)
    y_pred = fm.predict(X)
    print y_pred
    assert metrics.accuracy_score(y, y_pred) > 0.95
    # check different size
    fm.fit(X[:2,], y[:2])


if __name__ == '__main__':
    test_fm_classification()
