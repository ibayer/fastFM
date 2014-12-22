import numpy as np
import scipy.sparse as sp
from sklearn import metrics
from fastFM import mcmc
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
    y_pred = fm.fit_predict(X, y_labels, X)

    fpr, tpr, thresholds = metrics.roc_curve(y_labels, y_pred)
    auc = metrics.auc(fpr, tpr)
    assert auc > 0.95
    y_pred = fm.predict(X[:2,])


def _test_mcmc_warm_start():
    X, y, coef = make_user_item_regression(label_stdev=0)
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=43)
    X_train = sp.csc_matrix(X_train)
    X_test = sp.csc_matrix(X_test)

    fm = mcmc.FMRegression(n_iter=10, rank=2)
    y_pred = fm.fit_predict(X_train, y_train, X_test)
    error_10_iter = mean_squared_error(y_pred, y_test)

    fm = mcmc.FMRegression(n_iter=5, rank=2)
    y_pred = fm.fit_predict(X_train, y_train, X_test)
    error_5_iter = mean_squared_error(y_pred, y_test)

    y_pred = fm.fit_predict(X_train, y_train, X_test, n_more_iter=50)
    error_5_iter_plus_5 = mean_squared_error(y_pred, y_test)
    print error_5_iter, error_5_iter_plus_5, error_10_iter
    print fm.hyper_param_
    assert_almost_equal(error_10_iter, error_5_iter_plus_5, decimal=2)


if __name__ == "__main__":
    #test_mcmc_warm_start()

    X, y, coef = make_user_item_regression(label_stdev=.4)
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    X_train = sp.csc_matrix(X_train)
    X_test = sp.csc_matrix(X_test)

    offset = '../../fastFM-notes/benchmarks/'
    train_path = offset + "data/ml-100k/u1.base.libfm"
    test_path = offset + "data/ml-100k/u1.test.libfm"
    test_path = train_path

    from sklearn.datasets import load_svmlight_file
    X_train, y_train = load_svmlight_file(train_path)
    X_test,  y_test= load_svmlight_file(test_path)
    X_train = sp.csc_matrix(X_train)
    X_test = sp.csc_matrix(X_test)
    # add padding for features not in test
    X_test = sp.hstack([X_test, sp.csc_matrix((X_test.shape[0], X_train.shape[1] - X_test.shape[1]))])

    n_iter = 300
    count = 0

    fm = mcmc.FMRegression(n_iter=1, rank=8, random_state=50)
    # initalize coefs
    fm.fit_predict(X_train, y_train, X_test)

    rmse_test = []
    rmse_new = []
    hyper_param = np.zeros((n_iter, 5), dtype=np.float64)
    step_size = 1
    for i in range(n_iter):
        y_pred = fm.fit_predict(X_train, y_train, X_test, n_more_iter=step_size)
        rmse_test.append(np.sqrt(mean_squared_error(y_pred, y_test)))
        hyper_param[count, :] = fm.hyper_param_
        count = count + 1


    count = 0
    values = [1, 5, 10, 20, 30, 50, 100, 150]
    rmse_test2 = []
    #hyper_param = np.zeros((n_iter, 5), dtype=np.float64)
    for i in values:
        fm = mcmc.FMRegression(n_iter=i, rank=8, random_state=50)
        y_pred = fm.fit_predict(X_train, y_train, X_test)
        rmse_test2.append(np.sqrt(mean_squared_error(y_pred, y_test)))
        #hyper_param[count, :] = fm.hyper_param_
        count = count + 1

    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(nrows=4, sharex=True, figsize=(15, 10))

    x = np.arange(n_iter) * step_size
    #x = values


    #with plt.style.context('ggplot'):
    axes[0].plot(x, rmse_test, label='rmse test')
    axes[0].plot(values, rmse_test2, label='rmse restart')
    axes[0].legend()

    axes[1].plot(x, hyper_param[:,0], label='alpha')
    axes[1].legend()

    axes[2].plot(x, hyper_param[:,1], label='lambda_w')
    axes[2].plot(x, hyper_param[:,2], label='lambda_V')
    axes[2].legend()

    axes[3].plot(x, hyper_param[:,3], label='mu_w')
    axes[3].plot(x, hyper_param[:,4], label='mu_V')
    axes[3].legend()

    plt.show()
