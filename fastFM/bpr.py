from sklearn.utils import assert_all_finite
from base import FactorizationMachine
import ffm



class FMRecommender(FactorizationMachine):

    """ Factorization Machine Recommender with pairwise (BPR) loss solver.

    Parameters
    ----------
    max_iter : int, optional
        The number of samples for the MCMC sampler, number or iterations over the
        training set for ALS and number of steps for SGD.

    init_var: float, optional
        Sets the variance for the initialization of the parameter

    random_state: int, optional
        The seed of the pseudo random number generator that
        initializes the parameters and mcmc chain.

    rank: int
        The rank of the factorization used for the second order interactions.

    l2_reg_w : float
        L2 penalty weight for pairwise coefficients.

    l2_reg_V : float
        L2 penalty weight for linear coefficients.

    step_size : float
        Stepsize for the SGD solver, the solver uses a fixed step size and
        might require a tunning of the number of iterations `max_iter`.

    Attributes
    ---------

    w0_ : float
        bias term

    w_ : float | array, shape = (n_features)
        Coefficients for linear combination.

    V_ : float | array, shape = (rank_pair, n_features)
        Coefficients of second order factor matrix.
    """

    def __init__(self, max_iter=100, init_var=0.1, rank=8, random_state=123,
            l2_reg_w=0, l2_reg_V=0, step_size=0.1):
        super(FMRecommender, self).__init__(max_iter=max_iter,
            init_var=init_var, rank=rank, random_state=random_state)
        self.l2_reg_w = l2_reg_w
        self.l2_reg_V = l2_reg_V
        self.step_size = step_size
        self.task = "ranking"


    def fit(self, X_train, pairs):
        """ Fit model with specified loss.

        Parameters
        ----------
        X : scipy.sparse.csc_matrix, (n_samples, n_features)

        y : float | ndarray, shape = (n_compares, 2)
                Each row `i` defines a pair of samples such that
                the first returns a high value then the second
                FM(X[i,0]) > FM(X[i, 1]).
        """
        assert_all_finite(X_train)
        assert_all_finite(pairs)
        assert pairs.max() <= X_train.shape[1]
        assert pairs.min() >= 0
        self.w0_, self.w_, self.V_ = ffm.ffm_fit_sgd_bpr(self,
                                                X_train, pairs)
        return self
