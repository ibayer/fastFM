# Author: Immanuel Bayer
# License: BSD 3 clause

cimport cpp_ffm
from cpp_ffm cimport Settings, Data, Model, predict
from libcpp.memory cimport nullptr

cimport numpy as np
import numpy as np

def ffm2_predict(double w_0, double[:] w,
                np.ndarray[np.float64_t, ndim = 2] V, X):
    #assert X.shape[1] == len(w)
    #assert X.shape[1] == V.shape[1]

    # get attributes from csc scipy
    n_features = X.shape[1]
    n_samples = X.shape[0]
    nnz = X.count_nonzero()
    cdef np.ndarray[int, ndim=1, mode='c'] outer = X.indices
    cdef np.ndarray[int, ndim=1, mode='c'] inner = X.indptr
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
