from sklearn.base import RegressorMixin
from sklearn.utils import assert_all_finite
from base import FactorizationMachine, BaseFMClassifier, _validate_class_labels
import ffm


class FMRegression(FactorizationMachine, RegressorMixin):


    def __init__(self, max_iter=100, init_var=0.1, rank=8, random_state=123,
            l2_reg_w=0, l2_reg_V=0, step_size=0.1):
        super(FMRegression, self).__init__(max_iter=max_iter,
            init_var=init_var, rank=rank, random_state=random_state)
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


class FMClassification(BaseFMClassifier):


    def __init__(self, max_iter=100, init_var=0.1, rank=8, random_state=123,
            l2_reg_w=0, l2_reg_V=0, step_size=0.1):
        super(FMClassification, self).__init__(max_iter=max_iter,
            init_var=init_var, rank=rank, random_state=random_state)
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
