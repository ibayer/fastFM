# Author: Immanuel Bayer
# License: BSD 3 clause

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import mean_squared_error, r2_score

from .validation import check_random_state
from ffm import ffm_predict


def make_user_item_regression(random_state=123, n_user=20, n_item=20,
                              label_stdev=0.4, rank=2, bias=True,
                              first_order=True, stdev_w0=.2, stdev_w=0.3,
                              stdev_V=0.4, mean_w0=2, mean_w=5, mean_V=10):

    n_features = n_user + n_item
    n_samples = n_user * n_item
    # create design matrix
    user_cols = np.repeat(range(n_user), n_item)
    item_cols = np.array(list(range(n_item)) * n_user) + n_user
    cols = np.hstack((user_cols, item_cols))
    rows = np.hstack((np.arange(n_item*n_user), np.arange(n_item*n_user)))

    X = sp.coo_matrix((np.ones_like(cols, dtype=np.float64), (rows, cols)))
    X = sp.csc_matrix(X)
    assert X.shape[0] == n_samples
    assert X.shape[1] == n_features

    # sample the model parameter
    random_state = check_random_state(random_state)
    w0 = random_state.normal(mean_w0, stdev_w0)
    w = random_state.normal(mean_w, stdev_w, n_features)
    V = random_state.normal(mean_V, stdev_V, (rank, n_features))

    y = ffm_predict(w0, w, V, X)
    if label_stdev > 0:
        y = random_state.normal(y, label_stdev)

    return X, y, (w0, w, V)


if __name__ == '__main__':
    X, y, coef = make_user_item_regression(n_user=5, n_item=5, rank=2,
                                           label_stdev=2)
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    from mcmc import FMRegression
    fm = FMRegression(rank=2)
    y_pred = fm.fit_predict(sp.csc_matrix(X_train), y_train,
                            sp.csc_matrix(X_test))

    print('rmse', mean_squared_error(y_pred, y_test))
    print('r2_score', r2_score(y_pred, y_test))
    np.random.shuffle(y_pred)
    print('----  shuffled pred ---------')
    print('rmse', mean_squared_error(y_pred, y_test))
    print('r2_score', r2_score(y_pred, y_test))
