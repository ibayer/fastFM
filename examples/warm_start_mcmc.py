from fastFM.datasets import make_user_item_regression
from fastFM import mcmc
from sklearn.metrics import mean_squared_error
import scipy.sparse as sp
import numpy as np


if __name__ == "__main__":


    offset = '../../fastFM-notes/benchmarks/'
    train_path = offset + "data/ml-100k/u1.base.libfm"
    test_path = offset + "data/ml-100k/u1.test.libfm"
    #test_path = train_path

    from sklearn.datasets import load_svmlight_file
    X_train, y_train = load_svmlight_file(train_path)
    X_test,  y_test= load_svmlight_file(test_path)
    X_train = sp.csc_matrix(X_train)
    X_test = sp.csc_matrix(X_test)
    # add padding for features not in test
    X_test = sp.hstack([X_test, sp.csc_matrix((X_test.shape[0], X_train.shape[1] - X_test.shape[1]))])

    """
    X_train = sp.csc_matrix(np.array([[6, 1],
                                [2, 3],
                                [3, 0],
                                [6, 1],
                                [4, 5]]), dtype=np.float64)
    y_train = np.array([298, 266, 29, 298, 848], dtype=np.float64)
    X_test = X_train
    y_test = y_train
    """

    n_iter = 50
    rank = 4
    seed = 333
    step_size = 1

    """
    X, y, coef = make_user_item_regression(label_stdev=.4, random_state=seed)
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=seed)
    X_train = sp.csc_matrix(X_train)
    X_test = sp.csc_matrix(X_test)
    X_test = X_train
    y_test = y_train
    """

    fm = mcmc.FMRegression(n_iter=0, rank=rank, random_state=seed)
    # initalize coefs
    fm.fit_predict(X_train, y_train, X_test)

    rmse_test = []
    rmse_new = []
    hyper_param = np.zeros((n_iter -1, 3 + 2 * rank), dtype=np.float64)
    for nr, i in enumerate(range(1, n_iter)):
        fm.random_state = i * seed
        y_pred = fm.fit_predict(X_train, y_train, X_test, n_more_iter=step_size)
        rmse_test.append(np.sqrt(mean_squared_error(y_pred, y_test)))
        hyper_param[nr, :] = fm.hyper_param_

    print '------- restart ----------'
    values = np.arange(1, n_iter)
    rmse_test_re = []
    hyper_param_re = np.zeros((len(values), 3 + 2 * rank), dtype=np.float64)
    for nr, i in enumerate(values):
        fm = mcmc.FMRegression(n_iter=i, rank=rank, random_state=seed)
        y_pred = fm.fit_predict(X_train, y_train, X_test)
        rmse_test_re.append(np.sqrt(mean_squared_error(y_pred, y_test)))
        hyper_param_re[nr, :] = fm.hyper_param_

    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(15, 8))

    x = values * step_size
    burn_in = 5
    x = x[burn_in:]

    #with plt.style.context('ggplot'):
    axes[0, 0].plot(x, rmse_test[burn_in:], label='test rmse', color="r")
    axes[0, 0].plot(values[burn_in:], rmse_test_re[burn_in:], ls="--", color="r")
    axes[0, 0].legend()

    axes[0, 1].plot(x, hyper_param[burn_in:,0], label='alpha', color="b")
    axes[0, 1].plot(values[burn_in:], hyper_param_re[burn_in:,0], ls="--", color="b")
    axes[0, 1].legend()

    axes[1, 0].plot(x, hyper_param[burn_in:,1], label='lambda_w', color="g")
    #axes[2].plot(x, hyper_param[:,2], label='lambda_V', color="r")
    axes[1, 0].plot(values[burn_in:], hyper_param_re[burn_in:,1], ls="--", color="g")
    #axes[2].plot(values, hyper_param_re[:,2], label='lambda_V', ls="--", color="r")
    axes[1, 0].legend()

    axes[1, 1].plot(x, hyper_param[burn_in:,3], label='mu_w', color="g")
    #axes[3].plot(x, hyper_param[:,4], label='mu_V', color="r")
    axes[1, 1].plot(values[burn_in:], hyper_param_re[burn_in:,3], ls="--", color="g")
    #axes[3].plot(values, hyper_param_re[:,4], label='mu_V', ls="--", color="r")
    axes[1, 1].legend()

    plt.show()
    #plt.savefig("../../fastFM-notes/jmlr/figs/mcmc_trace.pdf", bbox_inches='tight')
