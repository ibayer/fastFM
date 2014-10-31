import numpy as np
import ffm
import scipy.sparse as sp
from scipy.stats import norm
from sklearn.utils import assert_all_finite


class FactorizationMachine:
    """Linear model combined with factorized coefficients for second order
    interactions between features.

    Parameters
    ----------
    max_iter : int, optional
        The number of samples for the MCMC sampler, number or iterations over the
        training set for ALS and number of steps for SGD.

    random_state: int, optional
        The seed of the pseudo random number generator that
        initializes the parameters and mcmc chain.

    init_stdev : float, optional
        Sets the variance for the initialization of the parameter 
        factorization

    solver : 'mcmc' | 'sgd'
        Selects the solver, note that for ranking (BPR) only `sgd` is
        implemented.

    task : 'regression' | 'classification' | 'ranking'
        Specifies the loss function, l2 loss for `regression`, sigmoid for
        `classification' and BPR for `ranking`.

    step_size : float
        Stepsize for the SGD solver, the solver uses a fixed step size and
        might require a tunning of the number of iterations `max_iter`.

    lambda_V : float
        L2 penalty weight for pairwise coefficients.

    lambda_w : float
        L2 penalty weight for linear coefficients.

    rank_pair: int
        The rank of the factorization used for the second order interactions.

    Attributes
    ---------

    w0_ : float
        bias term

    w_ : float | array, shape = (n_features)
        Coefficients for linear combination.

    V_ : float | array, shape = (rank_pair, n_features)
        Coefficients of second order factor matrix.
    """

    def __init__(self, max_iter=100, init_stdev=0.1, solver='mcmc',
            task='regression', rank_pair=0, lambda_V=1, lambda_w=1,
            step_size=0.1, random_state=123):
        self.max_iter = max_iter
        self.random_state = random_state
        self.init_stdev = init_stdev
        self.solver = solver
        self.task = task
        self.step_size = step_size 
        self.lambda_V = lambda_V
        self.lambda_w = lambda_w
        self.rank_pair = rank_pair
        self.w0_ = None
        self.w_ = None
        self.V_ = None

    def fit(self, X_train, y_train):
        """ Fit model with specified loss.

        Parameters
        ----------
        X : scipy.sparse.csc_matrix, (n_samples, n_features)

        y : float | ndarray, shape = (n_samples, )

        """
        assert_all_finite(X_train)
        assert_all_finite(y_train)
        if (self.task in ['classification', 'regression']):
            self._fit(X_train, y_train)
        elif (self.task=='ranking'):
            assert y_train.max() <= X_train.shape[1]
            self.w0_, self.w_, self.V_ = ffm.ffm_fit_sgd_bpr(self,
                                                    X_train, y_train)
        else:
            raise Exception("task unknown")

    def _fit(self, X_train, y_train):
        if self.task == 'classification':
            assert len(set(y_train)) == 2
            assert y_train.min() == -1
            assert y_train.max() == 1
            assert sp.isspmatrix_csc(X_train)
        if self.solver == 'als':
            self.w0_, self.w_, self.V_ = ffm.ffm_als_fit(self, X_train, y_train)
        elif self.solver=='sgd':
            self.w0_, self.w_, self.V_ = ffm.ffm_sgd_fit(self, X_train, y_train)
        elif self.solver=='mcmc':
            raise Exception("mcmc can only be used with fit_predict")
        else:
            raise Exception("solver not implemented")


    def predict(self, X_test):
        """ Return predictions

        Parameters
        ----------
        X : scipy.sparse.csc_matrix, (n_samples, n_features)

        Returns
        ------

        T : array, shape (n_samples)
            The labels are returned for classification.

        """
        assert_all_finite(X_test)
        assert sp.isspmatrix_csc(X_test)
        assert X_test.shape[1] == len(self.w_)
        pred = ffm.ffm_predict(self.w0_, self.w_, self.V_, X_test)
        if self.task == 'regression':
            return pred
        if self.task == 'ranking':
            print pred
            return np.argsort(pred)
        y_pred = norm.cdf(pred)
        # convert probs to labels
        y_pred[y_pred < 0.5] = -1
        y_pred[y_pred >= 0.5] = 1
        return y_pred


    def predict_proba(self, X_test):
        """ Return probabilities

        Parameters
        ----------
        X : scipy.sparse.csc_matrix, (n_samples, n_features)

        Returns
        ------

        T : array, shape (n_samples)
            Class Probabilities

        """
        assert_all_finite(X_test)
        assert sp.isspmatrix_csc(X_test)
        if self.task == 'regression':
            raise Exception("Regression model can't return probabilities")
        return norm.cdf(ffm.ffm_predict(self.w0_, self.w_, self.V_, X_test))

    def fit_predict(self, X_train, y_train, X_test):
        """Return average of posterior estimates of the test samples.
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
        if self.task == 'classification':
            assert len(set(y_train)) == 2
            assert y_train.min() == -1
            assert y_train.max() == 1
        assert_all_finite(X_train)
        assert_all_finite(X_test)
        assert_all_finite(y_train)
        assert sp.isspmatrix_csc(X_test)
        assert X_train.shape[1] == X_test.shape[1]
        assert X_train.shape[0] == len(y_train)
        if (self.solver=='mcmc'):
            coef, y_pred = ffm.ffm_mcmc_fit_predict(self, X_train,
                    X_test, y_train)
            self.w0_, self.w_, self.V_ = coef
            return y_pred
        else:
            raise Exception("use only with mcmc")
