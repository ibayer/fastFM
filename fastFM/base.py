# Author: Immanuel Bayer
# License: BSD 3 clause

import numpy as np
import scipy.sparse as sp
from scipy.stats import norm
from scipy.special import expit as sigmoid
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state

import ffm2
from .validation import check_array


def _init_parameter(fm, n_features):
    generator = check_random_state(fm.random_state)
    w0 = np.zeros(1, dtype=np.float64)
    w = np.zeros(n_features, dtype=np.float64)
    V = generator.normal(loc=0.0, scale=fm.init_stdev,
                         size=(fm.rank, n_features))
    return w0, w, V


def _settings_factory(fm):
    settings_dict = fm.get_params()
    settings_dict['loss'] = fm.loss
    settings_dict['solver'] = fm.solver

    # TODO align naming
    settings_dict['iter'] = int(settings_dict['n_iter'])
    del settings_dict['n_iter']

    return settings_dict


def _validate_class_labels(y):
    assert len(set(y)) == 2
    assert y.min() == -1
    assert y.max() == 1
    return check_array(y, ensure_2d=False, dtype=np.float64)


def _check_warm_start(fm, X):
    n_features = X.shape[1]
    if not fm.ignore_w_0:
        assert fm.w0_ is not None
    if not fm.ignore_w:
        assert fm.w_ is not None
        assert fm.w_.shape[0] == n_features
    if not fm.rank == 0:
        assert fm.V_.shape[1] == n_features


class FactorizationMachine(BaseEstimator):

    """ Factorization Machine trained MCMC (Gibbs) sampling.
    The predictions need to be calculated at training time since the individual
    parameter samples are to expensive to store.

    Parameters
    ----------
    n_iter : int, optional
        The number of samples for the MCMC sampler, number or iterations over
        the training set for ALS and number of steps for SGD.

    init_stdev: float, optional
        Sets the stdev for the initialization of the parameter

    random_state: int, optional
        The seed of the pseudo random number generator that
        initializes the parameters and mcmc chain.

    rank: int
        The rank of the factorization used for the second order interactions.

    copy_X : boolean, optional, default True
            If ``True``, X will be copied; else, it may be overwritten.

    Attributes
    ---------
    Attention these Coefficients are the last sample from the MCMC chain
    and can't be used to calculate predictions.

    w0_ : float
        bias term

    w_ : float | array, shape = (n_features)
        Coefficients for linear combination.

    V_ : float | array, shape = (rank_pair, n_features)
        Coefficients of second order factor matrix.
    """
    def __init__(self, n_iter=100, init_stdev=0.1, rank=8, random_state=123,
                 copy_X=True):
        self.n_iter = n_iter
        self.random_state = random_state
        self.init_stdev = init_stdev
        self.rank = rank
        self.iter_count = 0
        self.warm_start = False
        self.ignore_w_0 = False
        self.ignore_w = False
        self.l2_reg_w = 0.1
        self.l2_reg_V = 0.1
        self.step_size = 0
        self.copy_X = copy_X


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
        X_test = check_array(X_test, accept_sparse="csc", dtype=np.float64,
                             order="F")
        assert sp.isspmatrix_csc(X_test)
        assert X_test.shape[1] == len(self.w_)
        return ffm2.ffm_predict(self.w0_, self.w_, self.V_, X_test)


class BaseFMClassifier(FactorizationMachine, ClassifierMixin):

    def predict(self, X_test, threshold=0.5):
        """ Return predictions

        Parameters
        ----------
        X : scipy.sparse.csc_matrix, (n_samples, n_features)

        Returns
        ------

        y : array, shape (n_samples)
            Class labels
        """

        if self.loss == "logistic":
            y_proba = self.predict_proba(X_test)
            y_binary = np.ones_like(y_proba, dtype=np.float64)
            y_binary[y_proba < threshold] = -1
            return y_binary

        y_proba = norm.cdf(super(BaseFMClassifier, self).predict(X_test))
        # convert probs to labels
        y_pred = np.zeros_like(y_proba, dtype=np.float64) + self.classes_[0]
        y_pred[y_proba > .5] = self.classes_[1]
        return y_pred

    def predict_proba(self, X_test):
        """ Return probabilities

        Parameters
        ----------
        X : scipy.sparse.csc_matrix, (n_samples, n_features)

        Returns
        ------

        y : array, shape (n_samples)
            Class Probability for the class with smaller label.
        """

        if self.loss == "logistic":
            pred = ffm2.ffm_predict(self.w0_, self.w_, self.V_, X_test)
            return sigmoid(pred)

        pred = super(BaseFMClassifier, self).predict(X_test)
        return norm.cdf(pred)
