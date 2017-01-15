# Author: Immanuel Bayer
# License: BSD 3 clause

cimport cffm
from cffm cimport cs_di, ffm_param
# Import some functionality from Python and the C stdlib
from cpython.pycapsule cimport *

from libc.stdlib cimport malloc, free
from scipy.sparse import csc_matrix
cimport numpy as np
import numpy as np


# Destructor for cleaning up CsMatrix objects
cdef del_CsMatrix(object obj):
    pt = <cffm.cs_di *> PyCapsule_GetPointer(obj, "CsMatrix")
    free(<void *> pt)


# Create a CsMatrix object and return as a capsule
def CsMatrix(X not None):
    cdef cffm.cs_di *p
    p = <cffm.cs_di *> malloc(sizeof(cffm.cs_di))
    if p == NULL:
        raise MemoryError("No memory to make a Point")

    cdef int i
    cdef np.ndarray[int, ndim=1, mode = 'c'] indptr = X.indptr
    cdef np.ndarray[int, ndim=1, mode = 'c'] indices = X.indices
    cdef np.ndarray[double, ndim=1, mode = 'c'] data = X.data

    # Put the scipy data into the CSparse struct. This is just copying some
    # pointers.
    p.nzmax = X.data.shape[0]
    p.m = X.shape[0]
    p.n = X.shape[1]
    p.p = &indptr[0]
    p.i = &indices[0]
    p.x = &data[0]
    p.nz = -1  # to indicate CSC format
    return PyCapsule_New(<void *>p, "CsMatrix",
                         <PyCapsule_Destructor>del_CsMatrix)


# Destructor for cleaning up FFMParam objects
cdef del_FFMParam(object obj):
    pt = <cffm.ffm_param *> PyCapsule_GetPointer(obj, "FFMParam")
    free(<void *> pt)


# Create a FFMParam object and return as a capsule
def FFMParam(fm):
    map_flags = {'classification': 10,
                 'regression': 20,
                 'ranking': 30}
    cdef cffm.ffm_param *p
    p = <cffm.ffm_param *> malloc(sizeof(cffm.ffm_param))
    if p == NULL:
        raise MemoryError("No memory to make a FFMParam")
    p.n_iter = fm.n_iter
    p.k = fm.rank
    p.stepsize = fm.step_size
    p.init_sigma = fm.init_stdev
    p.TASK = map_flags[fm.task]
    p.rng_seed = fm.random_state
    p.init_lambda_w = fm.l2_reg_w
    p.init_lambda_V = fm.l2_reg_V
    p.iter_count = fm.iter_count

    p.ignore_w_0 = 1 if fm.ignore_w_0 else 0
    p.ignore_w = 1 if fm.ignore_w else 0
    p.warm_start = 1 if fm.warm_start else 0
    return PyCapsule_New(<void *>p, "FFMParam",
                         <PyCapsule_Destructor>del_FFMParam)


def ffm_predict(double w_0, double[:] w,
                np.ndarray[np.float64_t, ndim = 2] V, X):
    assert X.shape[1] == len(w)
    assert X.shape[1] == V.shape[1]
    X_ = CsMatrix(X)
    k = V.shape[0]
    pt_X = <cffm.cs_di *> PyCapsule_GetPointer(X_, "CsMatrix")
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] y =\
         np.zeros(X.shape[0], dtype=np.float64)
    cffm.ffm_predict(&w_0, &w[0], <double *> V.data, pt_X, &y[0], k)
    return y


def ffm_als_fit(fm, X, double[:] y):
    assert X.shape[0] == len(y) # test shapes
    n_features = X.shape[1]
    X_ = CsMatrix(X)
    pt_X = <cffm.cs_di *> PyCapsule_GetPointer(X_, "CsMatrix")
    param = FFMParam(fm)
    pt_param = <cffm.ffm_param *> PyCapsule_GetPointer(param, "FFMParam")
    cdef double w_0
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] w
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] V

    if fm.warm_start:
        w_0 = 0 if fm.ignore_w_0 else fm.w0_
        w = np.zeros(n_features, dtype=np.float64) if fm.ignore_w else fm.w_
        V = np.zeros((fm.rank, n_features), dtype=np.float64)\
                if fm.rank == 0 else fm.V_
    else:
        w_0 = 0
        w = np.zeros(n_features, dtype=np.float64)
        V = np.zeros((fm.rank, n_features), dtype=np.float64)

    cffm.ffm_als_fit(&w_0, <double *> w.data, <double *> V.data,
                     pt_X, &y[0], pt_param)
    return w_0, w, V


def ffm_sgd_fit(fm, X, double[:] y):
    """
    The sgd solver expects a transposed design matrix in column major order
    (csc_matrix) Samples are stored in columns, this allows fast sample by
    sample access.
    """
    assert X.shape[1] == len(y) # test shapes
    n_features = X.shape[0]
    X_ = CsMatrix(X)
    pt_X = <cffm.cs_di *> PyCapsule_GetPointer(X_, "CsMatrix")
    param = FFMParam(fm)
    pt_param = <cffm.ffm_param *> PyCapsule_GetPointer(param, "FFMParam")

    # allocate the coefs
    cdef double w_0 = 0
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] w =\
         np.zeros(n_features, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] V =\
         np.zeros((fm.rank, n_features), dtype=np.float64)

    cffm.ffm_sgd_fit(&w_0, <double *> w.data, <double *> V.data,
                     pt_X, &y[0], pt_param)
    return w_0, w, V


def ffm_fit_sgd_bpr(fm, X, np.ndarray[np.float64_t, ndim=2, mode='c'] pairs):
    n_features = X.shape[0]
    X_ = CsMatrix(X)
    pt_X = <cffm.cs_di *> PyCapsule_GetPointer(X_, "CsMatrix")
    param = FFMParam(fm)
    pt_param = <cffm.ffm_param *> PyCapsule_GetPointer(param, "FFMParam")

    #allocate the coefs
    cdef double w_0 = 0
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] w =\
         np.zeros(n_features, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] V =\
         np.zeros((fm.rank, n_features), dtype=np.float64)

    cffm.ffm_sgd_bpr_fit(&w_0, <double *> w.data, <double *> V.data,
                         pt_X, <double *> pairs.data, pairs.shape[0], pt_param)
    return w_0, w, V


def ffm_mcmc_fit_predict(fm, X_train, X_test, double[:] y):
    assert X_train.shape[0] == len(y)
    assert X_train.shape[1] == X_test.shape[1]
    n_features = X_train.shape[1]
    param = FFMParam(fm)
    pt_param = <cffm.ffm_param *> PyCapsule_GetPointer(param, "FFMParam")
    X_train_ = CsMatrix(X_train)
    pt_X_train = <cffm.cs_di *> PyCapsule_GetPointer(X_train_, "CsMatrix")
    X_test_ = CsMatrix(X_test)
    pt_X_test = <cffm.cs_di *> PyCapsule_GetPointer(X_test_, "CsMatrix")

    cdef double w_0
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] w
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] V
    # allocate the results vector
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] y_pred =\
         np.zeros(X_test.shape[0], dtype=np.float64)

    if fm.warm_start:
        w_0 = 0 if fm.ignore_w_0 else fm.w0_
        w = np.zeros(n_features, dtype=np.float64) if fm.ignore_w else fm.w_
        V = np.zeros((fm.rank, n_features), dtype=np.float64)\
                if fm.rank == 0 else fm.V_
    else:
        w_0 = 0
        w = np.zeros(n_features, dtype=np.float64)
        V = np.zeros((fm.rank, n_features), dtype=np.float64)

    if fm.warm_start:
        y_pred = fm.prediction_
    else:
        y_pred = np.zeros(X_test.shape[0], dtype=np.float64)

    # allocate vector for hyperparameter
    w_groups = 1
    n_hyper_param = 1 + 2 * w_groups + 2 * fm.rank
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] hyper_param

    if fm.warm_start:
        hyper_param = fm.hyper_param_
    else:
        hyper_param = np.zeros(n_hyper_param, dtype=np.float64)
    pt_param.n_hyper_param = n_hyper_param
    pt_param.hyper_param = <double *> hyper_param.data

    cffm.ffm_mcmc_fit_predict(&w_0, <double *> w.data, <double *> V.data,
                              pt_X_train, pt_X_test,
                              &y[0], <double *> y_pred.data,
                              pt_param)
    fm.hyper_param_ = hyper_param
    return (w_0, w, V), y_pred


def cs_norm(X):
    X = CsMatrix(X)
    pt = <cffm.cs_di *> PyCapsule_GetPointer(X, "CsMatrix")
    return cffm.cs_di_norm(pt)
