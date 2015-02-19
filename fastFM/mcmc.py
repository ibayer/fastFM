from sklearn.utils import assert_all_finite
import scipy.sparse as sp
from base import FactorizationMachine, _validate_class_labels, _check_warm_start
import ffm
import numpy as np
from sklearn.metrics import mean_squared_error


def find_init_stdev(fm, X_train, y_train, X_vali=None, y_vali=None,
        stdev_range=None, ):
    if not stdev_range:
        stdev_range = [0.1, 0.1, 0.2, 0.5, 1.0]

    if not isinstance(fm, FMRegression):
        raise Exception("only implemented for FMRegression")

    # just using a dummy here
    if X_vali is None:
        X_test = X_train[:2, :]
    else: X_test = X_vali

    best_init_stdev = 0
    best_mse = np.finfo(np.float64).max
    for init_stdev in stdev_range:
        fm.init_stdev = init_stdev
        y_pred_vali = fm.fit_predict(X_train, y_train, X_test)
        if X_vali is None:
            y_pred = fm.predict(X_train)
            mse = mean_squared_error(y_pred, y_train)
        else: mse = mean_squared_error(y_pred_vali, y_vali)
        if mse < best_mse:
            best_mse = mse
            best_init_stdev = init_stdev
    return best_init_stdev, best_mse


def _validate_mcmc_fit_input(X_train, y_train, X_test):
        assert_all_finite(X_train)
        assert_all_finite(X_test)
        assert_all_finite(y_train)
        assert sp.isspmatrix_csc(X_test)
        assert X_train.shape[1] == X_test.shape[1]
        assert X_train.shape[0] == len(y_train)


class FMRegression(FactorizationMachine):


    def fit_predict(self, X_train, y_train, X_test, n_more_iter=0):
        """Return average of posterior estimates of the test samples.

        Parameters
        ----------
        X_train : scipy.sparse.csc_matrix, (n_samples, n_features)

        y_train : array, shape (n_samples)

        X_test : scipy.sparse.csc_matrix, (n_test_samples, n_features)

        n_more_iter : int
                Number of iterations to continue from the current Coefficients.

        Returns
        ------

        T : array, shape (n_test_samples)
        """
        self.task = "regression"
        _validate_mcmc_fit_input(X_train, y_train, X_test)

        self.n_iter = self.n_iter + n_more_iter

        if n_more_iter > 0:
            _check_warm_start(self, X_train)
            assert self.prediction_.shape[0] == X_test.shape[0]
            assert self.hyper_param_.shape
            self.warm_start = True
        else:
            self.iter_count = 0

        coef, y_pred = ffm.ffm_mcmc_fit_predict(self, X_train,
                X_test, y_train)
        self.w0_, self.w_, self.V_ = coef
        self.prediction_ = y_pred
        self.warm_start = False

        if self.iter_count != 0:
            self.iter_count = self.iter_count + n_more_iter
        else:
            self.iter_count = self.n_iter

        return y_pred


class FMClassification(FactorizationMachine):


    def fit_predict(self, X_train, y_train, X_test):
        """Return average class probabilities of posterior estimates of the test samples.
        Use only with MCMC!

        Parameters
        ----------
        X_train : scipy.sparse.csc_matrix, (n_samples, n_features)

        y_train : array, shape (n_samples)
                the targets have to be encodes as {-1, 1}.

        X_test : scipy.sparse.csc_matrix, (n_test_samples, n_features)

        Returns
        ------

        T : array, shape (n_test_samples)
        """
        self.task = "classification"
        _validate_mcmc_fit_input(X_train, y_train, X_test)
        _validate_class_labels(y_train)

        coef, y_pred = ffm.ffm_mcmc_fit_predict(self, X_train,
                X_test, y_train)
        self.w0_, self.w_, self.V_ = coef
        return y_pred
