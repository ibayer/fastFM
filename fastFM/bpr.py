# Author: Immanuel Bayer
# License: BSD 3 clause

import ffm
import numpy as np
from .base import FactorizationMachine
from sklearn.utils.testing import assert_array_equal
from .validation import check_array, assert_all_finite


class FMRecommender(FactorizationMachine):

    """ Factorization Machine Recommender with pairwise (BPR) loss solver.

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
        super(FMRecommender, self).\
            __init__(n_iter=n_iter, init_stdev=init_stdev, rank=rank,
                     random_state=random_state)
        if (l2_reg != 0):
            self.l2_reg_V = l2_reg
            self.l2_reg_w = l2_reg
        else:
            self.l2_reg_w = l2_reg_w
            self.l2_reg_V = l2_reg_V
        self.step_size = step_size
        self.task = "ranking"

    def fit(self, X, pairs):
        """ Fit model with specified loss.

        Parameters
        ----------
        X : scipy.sparse.csc_matrix, (n_samples, n_features)

        y : float | ndarray, shape = (n_compares, 2)
                Each row `i` defines a pair of samples such that
                the first returns a high value then the second
                FM(X[i,0]) > FM(X[i, 1]).
        """
        # The sgd solver expects a transposed design matrix in column major
        # order (csc_matrix).
        X = X.T  # creates a copy
        X = check_array(X, accept_sparse="csc", dtype=np.float64)
        assert_all_finite(pairs)

        pairs = pairs.astype(np.float64)

        # check that pairs contain no real values
        assert_array_equal(pairs, pairs.astype(np.int32))
        assert pairs.max() <= X.shape[1]
        assert pairs.min() >= 0
        self.w0_, self.w_, self.V_ = ffm.ffm_fit_sgd_bpr(self, X, pairs)
        return self
