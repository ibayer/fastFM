# Author: Immanuel Bayer
# License: BSD 3 clause

import numpy as np
import scipy.sparse as sp
from fastFM import bpr
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


def test_fm_sgr_ranking():
    w0, w, V, y, X = get_test_problem()
    X_test = X.copy()
    X_train = X.copy()

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

    print(compares)
    fm = bpr.FMRecommender(n_iter=2000,
                           init_stdev=0.01, l2_reg_w=.5, l2_reg_V=.5, rank=2,
                           step_size=.002, random_state=11)
    fm.fit(X_train, compares)
    y_pred = fm.predict(X_test)
    y_pred = np.argsort(y_pred)
    print(y)
    print(y_pred)
    print(np.argsort(y))
    assert utils.kendall_tau(np.argsort(y), y_pred) == 1
