# Author: Immanuel Bayer
# License: BSD 3 clause

import numpy as np
import scipy.sparse as sp
from sklearn import metrics
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


def test_fm_classification_predict_proba():
    w0, w, V, y, X = get_test_problem(task='classification')

    fm = als.FMClassification(n_iter=1000,
                              init_stdev=0.1, l2_reg_w=0, l2_reg_V=0, rank=2)
    fm.fit(X, y)
    y_pred = fm.predict(X)
    y_pred = fm.predict_proba(X)

    y[y == -1] = 0
    assert metrics.roc_auc_score(y, y_pred) > 0.95

if __name__ == '__main__':
    test_fm_classification_predict_proba()
