# Author: Immanuel Bayer
# License: BSD 3 clause

cimport cpp_ffm
from cpp_ffm cimport Settings, Data, Model, fit, predict
from libcpp.memory cimport nullptr

cimport numpy as np
import numpy as np

def ffm2_predict(double w_0, double[:] w,
                np.ndarray[np.float64_t, ndim = 2] V, X):
    assert X.shape[1] == len(w)
    assert X.shape[1] == V.shape[1]


    #X_ = CsMatrix(X)
    k = V.shape[0]
    #pt_X = <cffm.cs_di *> PyCapsule_GetPointer(X_, "CsMatrix")
    
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] y =\
         np.zeros(X.shape[0], dtype=np.float64)
    
    #cpp_ffm.ffm_predict(&w_0, &w[0], <double *> V.data, pt_X, &y[0], k)

    cdef int split = 0;
    cdef Model *m = new Model()
    cdef Data *d = new Data()
    
    m.add_parameter(&w_0)
    m.add_parameter(&w[0], k)

    d.add_prediction(&y[0], k, X.shape[0])
    #d.add_design_matrix()

    cpp_ffm.predict(m, d, split)

    del m
    del d

    # dummy return
    return w[0]
