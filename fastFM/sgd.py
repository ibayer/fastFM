# Author: Immanuel Bayer
# License: BSD 3 clause


import ffm
import numpy as np
from sklearn.base import RegressorMixin
from .validation import check_array, check_consistent_length
from .base import (FactorizationMachine, BaseFMClassifier,
                   _validate_class_labels)


class FMRegression(FactorizationMachine, RegressorMixin):

    """ Factorization Machine Regression trained with a stochastic gradient
    descent solver.

    Parameters
    ----------
    n_iter : int, optional
        The number of interations of individual samples .

    init_stdev: float, optional
        Sets the stdev for the initialization of the parameter

    random_state: int, optional
        The seed of the pseudo random number generator that
        initializes the parameters and mcmc chain.

    rank: int
        The rank of the factorization used for the second order interactions.

    l2_reg_w : float
        L2 penalty weight for linear coefficients.

    l2_reg_V : float
        L2 penalty weight for pairwise coefficients.

    l2_reg : float
        L2 penalty weight for all coefficients (default=0).

    step_size : float
        Stepsize for the SGD solver, the solver uses a fixed step size and
        might require a tunning of the number of iterations `n_iter`.

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
                 l2_reg_w=0.1, l2_reg_V=0.1, l2_reg=0, step_size=0.1):
        super(FMRegression, self).\
            __init__(n_iter=n_iter, init_stdev=init_stdev, rank=rank,
                     random_state=random_state)
        if (l2_reg != 0):
            self.l2_reg_V = l2_reg
            self.l2_reg_w = l2_reg
        else:
            self.l2_reg_w = l2_reg_w
            self.l2_reg_V = l2_reg_V
        self.l2_reg = l2_reg
        self.step_size = step_size
        self.task = "regression"

    def fit(self, X, y):
        """ Fit model with specified loss.

        Parameters
        ----------
        X : scipy.sparse.csc_matrix, (n_samples, n_features)

        y : float | ndarray, shape = (n_samples, )

        """

        check_consistent_length(X, y)
        y = check_array(y, ensure_2d=False, dtype=np.float64)

        # The sgd solver expects a transposed design matrix in column major
        # order (csc_matrix).
        X = X.T  # creates a copy
        X = check_array(X, accept_sparse="csc", dtype=np.float64)

        self.w0_, self.w_, self.V_ = ffm.ffm_sgd_fit(self, X, y)
        return self


class FMClassification(BaseFMClassifier):

    """ Factorization Machine Classification trained with a stochastic gradient
    descent solver.

    Parameters
    ----------
    n_iter : int, optional
        The number of interations of individual samples .

    init_std: float, optional
        Sets the stdev for the initialization of the parameter

    random_state: int, optional
        The seed of the pseudo random number generator that
        initializes the parameters and mcmc chain.

    rank: int
        The rank of the factorization used for the second order interactions.

    l2_reg_w : float
        L2 penalty weight for linear coefficients.

    l2_reg_V : float
        L2 penalty weight for pairwise coefficients.

    l2_reg : float
        L2 penalty weight for all coefficients (default=0).

    step_size : float
        Stepsize for the SGD solver, the solver uses a fixed step size and
        might require a tunning of the number of iterations `n_iter`.

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
                 l2_reg_w=0, l2_reg_V=0, l2_reg=None, step_size=0.1):
        super(FMClassification, self).\
            __init__(n_iter=n_iter, init_stdev=init_stdev, rank=rank,
                     random_state=random_state)
        if (l2_reg is not None):
            self.l2_reg_V = l2_reg
            self.l2_reg_w = l2_reg
        else:
            self.l2_reg_w = l2_reg_w
            self.l2_reg_V = l2_reg_V
        self.l2_reg = l2_reg
        self.step_size = step_size
        self.task = "classification"

    def fit(self, X, y):
        """ Fit model with specified loss.

        Parameters
        ----------
        X : scipy.sparse.csc_matrix, (n_samples, n_features)

        y : float | ndarray, shape = (n_samples, )

                the targets have to be encodes as {-1, 1}.
        """
        y = _validate_class_labels(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("This solver only supports binary classification"
                             " but the data contains"
                             " class: %r" % self.classes_)

        # fastFM-core expects labels to be in {-1,1}
        y_train = y.copy()
        i_class1 = (y_train == self.classes_[0])
        y_train[i_class1] = -1
        y_train[-i_class1] = 1

        check_consistent_length(X, y)
        y = y.astype(np.float64)

        # The sgd solver expects a transposed design matrix in column major
        # order (csc_matrix).
        X = X.T  # creates a copy
        X = check_array(X, accept_sparse="csc", dtype=np.float64)

        self.w0_, self.w_, self.V_ = ffm.ffm_sgd_fit(self, X, y)
        return self
