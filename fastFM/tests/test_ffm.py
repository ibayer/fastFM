import numpy as np
import scipy.sparse as sp
from sklearn.metrics import mean_squared_error, r2_score
from numpy.testing import assert_almost_equal, assert_equal
import ffm

def get_test_problem():
    X = sp.csc_matrix(np.array([[6, 1],
                                [2, 3],
                                [3, 0],
                                [6, 1],
                                [4, 5]]), dtype=np.float64)
    y = np.array([298, 266, 29, 298, 848], dtype=np.float64)
    V = np.array([[6, 0],
                  [5, 8]], dtype=np.float64)
    w = np.array([9, 2], dtype=np.float64)
    w0 = 2
    return w0, w, V, y, X

def test_cxsparse_integration():
    X = sp.csc_matrix(np.arange(60, dtype=np.float64).reshape(6,10))
    assert_almost_equal(ffm.cs_norm(X), X.sum(axis=0).max())

def test_ffm_vector():
    a = np.arange(10, dtype=np.float64)
    b = np.arange(10, dtype=np.float64) + 1
    #b = np.random.normal(1,1,10)
    assert_equal(mean_squared_error(a,b),
            ffm.ffm_mean_squared_error(a,b))

def test_ffm_matrix():
    X = np.arange(12, dtype=np.float64).reshape(3,4)
    assert_equal(X[2,3], ffm.ffm_matrix_get(X, 2, 3))
    assert_equal(X[2,0], ffm.ffm_matrix_get(X, 2, 0))

def test_ffm_predict():
    w0, w, V, y, X = get_test_problem()
    y_pred = ffm.ffm_predict(w0, w, V, X)
    assert_equal(y_pred, y)


if __name__ == '__main__':
    pass
