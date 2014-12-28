from sklearn.utils import assert_all_finite
import scipy.sparse as sp
from base import FactorizationMachine, _validate_class_labels, _check_warm_start
import ffm
import numpy as np


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
