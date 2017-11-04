# Author: Immanuel Bayer
# License: BSD 3 clause

cimport cpp_ffm
from cpp_ffm cimport Settings, Data, Model, predict, fit
from libcpp.memory cimport nullptr

cimport numpy as np
import numpy as np

def ffm_predict(double w_0, double[:] w,
                np.ndarray[np.float64_t, ndim = 2] V, X):
    assert X.shape[1] == len(w)
    assert X.shape[1] == V.shape[1]

    # get attributes from csc scipy
    n_features = X.shape[1]
    n_samples = X.shape[0]
    nnz = X.count_nonzero()

    cdef np.ndarray[int, ndim=1, mode='c'] inner = X.indices
    cdef np.ndarray[int, ndim=1, mode='c'] outer = X.indptr
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] data = X.data

    assert n_features == w.shape[0]
    assert n_features == V.shape[1]

    rank = V.shape[0]

    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] y =\
         np.zeros(X.shape[0], dtype=np.float64)

    cdef Model *m = new Model()
    cdef Data *d = new Data()

    m.add_parameter(&w_0)
    m.add_parameter(&w[0], n_features)
    m.add_parameter(<double *> V.data, rank, n_features, 2)

    d.add_target(n_samples, &y[0])
    d.add_prediction(n_samples, &y[0])
    d.add_design_matrix(n_samples, n_features, nnz, &outer[0], &inner[0],
                        &data[0])

    cpp_ffm.predict(m, d)

    del m
    del d

    return y

def ffm_als_fit(fm, X, double[:] y):
    assert X.shape[0] == len(y) # test shapes

    n_features = X.shape[1]
    n_samples = X.shape[0]
    nnz = X.count_nonzero()

    cdef np.ndarray[int, ndim=1, mode='c'] inner = X.indices
    cdef np.ndarray[int, ndim=1, mode='c'] outer = X.indptr
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] data = X.data
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] y_pred = np.zeros(n_samples, dtype=np.float64)

    cdef Data* d = new Data()
    d.add_design_matrix(n_samples, n_features, nnz, &outer[0], &inner[0], &data[0])
    d.add_target(n_samples, &y[0])
    d.add_prediction(n_samples, &y_pred[0])

    cdef Model* m = new Model()

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

    m.add_parameter(&w_0)
    m.add_parameter(&w[0], n_features)
    m.add_parameter(<double *> V.data, fm.rank, n_features, 2)


    cdef Settings* s = new Settings()

    cpp_ffm.fit(s, m, d)

    del d
    del m
    del s

    return w_0, w, V