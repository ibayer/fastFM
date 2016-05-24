# Author: Immanuel Bayer
# License: BSD 3 clause

import ffm
import numpy as np
from sklearn.base import RegressorMixin
from .validation import check_consistent_length, check_array
from .base import (FactorizationMachine, BaseFMClassifier,
                   _validate_class_labels, _check_warm_start)


class FMRegression(FactorizationMachine, RegressorMixin):

    """ Factorization Machine Regression trained with a als (coordinate descent)
    solver.

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

    l2_reg_w : float
        L2 penalty weight for pairwise coefficients.

    l2_reg_V : float
        L2 penalty weight for linear coefficients.

    l2_reg : float
        L2 penalty weight for all coefficients (default=0).

    Attributes
    ---------

    w0_ : float
        bias term

    w_ : float | array, shape = (n_features)
        Coefficients for linear combination.

    V_ : float | array, shape = (rank_pair, n_features)
        Coefficients of second order factor matrix.
    """
    def __init__(self, n_iter=100, init_stdev=0.1, rank=8, random_state=123,
                 l2_reg_w=0.1, l2_reg_V=0.1, l2_reg=0):
        super(FMRegression, self).__init__(n_iter=n_iter,
                                           init_stdev=init_stdev, rank=rank,
                                           random_state=random_state)
        if (l2_reg != 0):
            self.l2_reg_V = l2_reg
            self.l2_reg_w = l2_reg
        else:
            self.l2_reg_w = l2_reg_w
            self.l2_reg_V = l2_reg_V
        self.task = "regression"

    def fit(self, X_train, y_train, n_more_iter=0):
        """ Fit model with specified loss.

        Parameters
        ----------
        X : scipy.sparse.csc_matrix, (n_samples, n_features)

        y : float | ndarray, shape = (n_samples, )

        n_more_iter : int
                Number of iterations to continue from the current Coefficients.

        """

        check_consistent_length(X_train, y_train)
        y_train = check_array(y_train, ensure_2d=False, dtype=np.float64)

        X_train = check_array(X_train, accept_sparse="csc", dtype=np.float64,
                              order="F")
        self.n_iter = self.n_iter + n_more_iter

        if n_more_iter > 0:
            _check_warm_start(self, X_train)
            self.warm_start = True

        self.w0_, self.w_, self.V_ = ffm.ffm_als_fit(self, X_train, y_train)

        if self.iter_count != 0:
            self.iter_count = self.iter_count + n_more_iter
        else:
            self.iter_count = self.n_iter

        # reset to default setting
        self.warm_start = False
        return self


class FMClassification(BaseFMClassifier):

    """ Factorization Machine Classification trained with a ALS
    (coordinate descent)
    solver.

    Parameters
    ----------
    n_iter : int, optional
        The number of samples for the MCMC sampler, number or iterations over
        the training set for ALS and number of steps for SGD.

    init_stdev: float, optional
        Sets the stdev  for the initialization of the parameter

    random_state: int, optional
        The seed of the pseudo random number generator that
        initializes the parameters and mcmc chain.

    rank: int
        The rank of the factorization used for the second order interactions.

    l2_reg_w : float
        L2 penalty weight for pairwise coefficients.

    l2_reg_V : float
        L2 penalty weight for linear coefficients.

    l2_reg : float
        L2 penalty weight for all coefficients (default=0).

    Attributes
    ---------

    w0_ : float
        bias term

    w_ : float | array, shape = (n_features)
        Coefficients for linear combination.

    V_ : float | array, shape = (rank_pair, n_features)
        Coefficients of second order factor matrix.
    """
    def __init__(self, n_iter=100, init_stdev=0.1, rank=8, random_state=123,
                 l2_reg_w=0.1, l2_reg_V=0.1, l2_reg=None):
        super(FMClassification, self).__init__(n_iter=n_iter,
                                               init_stdev=init_stdev,
                                               rank=rank,
                                               random_state=random_state)
        if (l2_reg is not None):
            self.l2_reg_V = l2_reg
            self.l2_reg_w = l2_reg
        else:
            self.l2_reg_w = l2_reg_w
            self.l2_reg_V = l2_reg_V
        self.task = "classification"

    def fit(self, X_train, y_train):
        """ Fit model with specified loss.

        Parameters
        ----------
        X : scipy.sparse.csc_matrix, (n_samples, n_features)

        y : float | ndarray, shape = (n_samples, )
                the targets have to be encodes as {-1, 1}.
        """
        check_consistent_length(X_train, y_train)
        X_train = check_array(X_train, accept_sparse="csc", dtype=np.float64,
                              order="F")
        y_train = _validate_class_labels(y_train)

        self.classes_ = np.unique(y_train)
        if len(self.classes_) != 2:
            raise ValueError("This solver only supports binary classification"
                             " but the data contains"
                             " class: %r" % self.classes_)

        # fastFM-core expects labels to be in {-1,1}
        y_train = y_train.copy()
        i_class1 = (y_train == self.classes_[0])
        y_train[i_class1] = -1
        y_train[-i_class1] = 1

        self.w0_, self.w_, self.V_ = ffm.ffm_als_fit(self, X_train, y_train)
        return self
