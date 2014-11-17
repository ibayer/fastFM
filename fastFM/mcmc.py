from sklearn.utils import assert_all_finite
import scipy.sparse as sp
from base import FactorizationMachine, _validate_class_labels
import ffm


def _validate_mcmc_fit_input(X_train, y_train, X_test):
        assert_all_finite(X_train)
        assert_all_finite(X_test)
        assert_all_finite(y_train)
        assert sp.isspmatrix_csc(X_test)
        assert X_train.shape[1] == X_test.shape[1]
        assert X_train.shape[0] == len(y_train)


class FMRegression(FactorizationMachine):


    def fit_predict(self, X_train, y_train, X_test):
        """Return average of posterior estimates of the test samples.

        Parameters
        ----------
        X_train : scipy.sparse.csc_matrix, (n_samples, n_features)

        y_train : array, shape (n_samples)

        X_test : scipy.sparse.csc_matrix, (n_test_samples, n_features)

        Returns
        ------

        T : array, shape (n_test_samples)
        """
        self.task = "regression"
        _validate_mcmc_fit_input(X_train, y_train, X_test)
        coef, y_pred = ffm.ffm_mcmc_fit_predict(self, X_train,
                X_test, y_train)
        self.w0_, self.w_, self.V_ = coef
        return y_pred


class FMClassification(FactorizationMachine):


    def fit_predict(self, X_train, y_train, X_test):
        """Return average class probabilities of posterior estimates of the test samples.
        Use only with MCMC!

        Parameters
        ----------
        X_train : scipy.sparse.csc_matrix, (n_samples, n_features)

        y_train : array, shape (n_samples)

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
