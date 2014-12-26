from fastFM.datasets import make_user_item_regression
from fastFM import mcmc
from sklearn.metrics import mean_squared_error
import scipy.sparse as sp
import numpy as np


if __name__ == "__main__":

    X, y, coef = make_user_item_regression(label_stdev=.4)
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=45)
    X_train = sp.csc_matrix(X_train)
    X_test = sp.csc_matrix(X_test)
    X_test = X_train
    y_test = y_train

    """
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

    n_iter = 300
    count = 0
    rank = 4
    seed = 333

    fm = mcmc.FMRegression(n_iter=1, rank=rank, random_state=seed)
    # initalize coefs
    fm.fit_predict(X_train, y_train, X_test)

    rmse_test = []
    rmse_new = []
    hyper_param = np.zeros((n_iter -1, 3 + 2 * rank), dtype=np.float64)
    step_size = 1
    for i in range(1, n_iter):
        y_pred = fm.fit_predict(X_train, y_train, X_test, n_more_iter=step_size)
        rmse_test.append(np.sqrt(mean_squared_error(y_pred, y_test)))
        hyper_param[count, :] = fm.hyper_param_
        count = count + 1

    print '------- restart ----------'
    count = 0
    values = [1, 2, 3, 4, 5, 10, 20, 30, 50, 100, 150]
    rmse_test2 = []
    hyper_param_re = np.zeros((len(values), 3 + 2 * rank), dtype=np.float64)
    for i in values:
        fm = mcmc.FMRegression(n_iter=i, rank=rank, random_state=seed)
        y_pred = fm.fit_predict(X_train, y_train, X_test)
        rmse_test2.append(np.sqrt(mean_squared_error(y_pred, y_test)))
        hyper_param_re[count, :] = fm.hyper_param_
        count = count + 1

    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(nrows=4, sharex=True, figsize=(15, 10))

    x = np.arange(n_iter-1) * step_size


    #with plt.style.context('ggplot'):
    axes[0].plot(x, rmse_test, label='rmse')
    axes[0].plot(values, rmse_test2, label='rmse restart')
    axes[0].legend()

    axes[1].plot(x, hyper_param[:,0], label='alpha')
    axes[1].plot(values, hyper_param_re[:,0], label='alpha re')
    axes[1].legend()

    axes[2].plot(x, hyper_param[:,1], label='lambda_w')
    axes[2].plot(x, hyper_param[:,2], label='lambda_V')
    axes[2].plot(values, hyper_param_re[:,1], label='lambda_w re')
    axes[2].plot(values, hyper_param_re[:,2], label='lambda_V re')
    axes[2].legend()

    axes[3].plot(x, hyper_param[:,3], label='mu_w')
    axes[3].plot(x, hyper_param[:,4], label='mu_V')
    axes[3].plot(values, hyper_param_re[:,3], label='mu_w re')
    axes[3].plot(values, hyper_param_re[:,4], label='mu_V re')
    axes[3].legend()

    plt.show()
