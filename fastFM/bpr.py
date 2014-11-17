from sklearn.utils import assert_all_finite
from base import FactorizationMachine
import ffm



class FMRecommender(FactorizationMachine):

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

        y : float | ndarray, shape = (n_samples, 2)

        """
        assert_all_finite(X_train)
        assert_all_finite(pairs)
        assert pairs.max() <= X_train.shape[1]
        assert pairs.min() >= 0
        self.w0_, self.w_, self.V_ = ffm.ffm_fit_sgd_bpr(self,
                                                X_train, pairs)
