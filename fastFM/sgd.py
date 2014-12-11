from sklearn.base import RegressorMixin
from sklearn.utils import assert_all_finite
from base import FactorizationMachine, BaseFMClassifier, _validate_class_labels
import ffm


class FMRegression(FactorizationMachine, RegressorMixin):

    """ Factorization Machine Regression trained with a stochastic gradient
    descent solver.

    Parameters
    ----------
    n_iter : int, optional
        The number of samples for the MCMC sampler, number or iterations over the
        training set for ALS and number of steps for SGD.

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
            l2_reg_w=0, l2_reg_V=0, l2_reg=0, step_size=0.1):
        super(FMRegression, self).__init__(n_iter=n_iter,
            init_stdev=init_stdev, rank=rank, random_state=random_state)
        if (l2_reg != 0):
            self.l2_reg_V = l2_reg
            self.l2_reg_w = l2_reg
        else:
            self.l2_reg_w = l2_reg_w
            self.l2_reg_V = l2_reg_V
        self.step_size = step_size
        self.task = "regression"


    def fit(self, X_train, y_train):
        """ Fit model with specified loss.

        Parameters
        ----------
        X : scipy.sparse.csc_matrix, (n_samples, n_features)

        y : float | ndarray, shape = (n_samples, )

        """
        assert_all_finite(X_train)
        assert_all_finite(y_train)

        self.w0_, self.w_, self.V_ = ffm.ffm_sgd_fit(self, X_train, y_train)
        return self


class FMClassification(BaseFMClassifier):

    """ Factorization Machine Classification trained with a stochastic gradient
    descent solver.

    Parameters
    ----------
    n_iter : int, optional
        The number of samples for the MCMC sampler, number or iterations over the
        training set for ALS and number of steps for SGD.

    init_std: float, optional
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
            l2_reg_w=0, l2_reg_V=0, l2_reg=0, step_size=0.1):
        super(FMClassification, self).__init__(n_iter=n_iter,
            init_stdev=init_stdev, rank=rank, random_state=random_state)
        if (l2_reg != 0):
            self.l2_reg_V = l2_reg
            self.l2_reg_w = l2_reg
        else:
            self.l2_reg_w = l2_reg_w
            self.l2_reg_V = l2_reg_V
        self.step_size = step_size
        self.task = "classification"


    def fit(self, X_train, y_train):
        """ Fit model with specified loss.

        Parameters
        ----------
        X : scipy.sparse.csc_matrix, (n_samples, n_features)

        y : float | ndarray, shape = (n_samples, )

                the targets have to be encodes as {-1, 1}.
        """
        _validate_class_labels(y_train)
        assert_all_finite(X_train)
        assert_all_finite(y_train)

        self.w0_, self.w_, self.V_ = ffm.ffm_sgd_fit(self, X_train, y_train)
        return self
