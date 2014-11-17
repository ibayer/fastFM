import numpy as np
import scipy.sparse as sp
from sklearn import metrics
from fastFM import mcmc


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

    fm = mcmc.FMRegression(max_iter=1000, rank=2, init_var=0.1)

    y_pred = fm.fit_predict(X, y, X)
    assert metrics.r2_score(y_pred, y) > 0.99


def test_fm_classification():
    w0, w, V, y, X = get_test_problem()
    # transform to labels easier problem then default one
    y_labels = np.ones_like(y)
    y_labels[y < np.mean(y)] = -1

    fm = mcmc.FMClassification(max_iter=1000, init_var=0.1, rank=2)
    y_pred = fm.fit_predict(X, y_labels, X)

    fpr, tpr, thresholds = metrics.roc_curve(y_labels, y_pred)
    auc = metrics.auc(fpr, tpr)
    assert auc > 0.95
    y_pred = fm.predict(X[:2,])
