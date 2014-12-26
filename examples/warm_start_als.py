from fastFM.datasets import make_user_item_regression
from fastFM import als
from sklearn.metrics import mean_squared_error
import scipy.sparse as sp
import numpy as np

if __name__ == "__main__":
    X, y, coef = make_user_item_regression(label_stdev=.4)
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    X_train = sp.csc_matrix(X_train)
    X_test = sp.csc_matrix(X_test)
    n_iter = 50
    results = np.zeros((n_iter, 2), dtype=np.float64)

    offset = '../../fastFM-notes/benchmarks/'
    train_path = offset + "data/ml-100k/u1.base.libfm"
    test_path = offset + "data/ml-100k/u1.test.libfm"

    from sklearn.datasets import load_svmlight_file
    X_train, y_train = load_svmlight_file(train_path)
    X_test,  y_test= load_svmlight_file(test_path)
    X_train = sp.csc_matrix(X_train)
    X_test = sp.csc_matrix(X_test)
    # add padding for features not in test
    X_test = sp.hstack([X_test, sp.csc_matrix((X_test.shape[0], X_train.shape[1] - X_test.shape[1]))])


    fm = als.FMRegression(n_iter=1, l2_reg_w=0, l2_reg_V=0, rank=2)
# initalize coefs
    fm.fit(X_train, y_train)

    rmse_train = []
    rmse_test = []
    for i in range(n_iter):
        fm.fit(X_train, y_train, warm_start=True)
        y_pred = fm.predict(X_test)
        rmse_train.append(mean_squared_error(fm.predict(X_train), y_train))
        rmse_test.append(mean_squared_error(fm.predict(X_test), y_test))

    from matplotlib import pyplot as plt

    x = np.arange(n_iter)
    with plt.style.context('fivethirtyeight'):
        plt.plot(x, rmse_train, label='train')
        plt.plot(x, rmse_test, label='test')
    plt.legend()
    plt.show()
