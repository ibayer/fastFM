# Author: Immanuel Bayer
# License: BSD 3 clause
#distutils: language=c++

from libcpp.string cimport string

cdef extern from "../fastFM-core2/fastFM/fastfm.h" namespace "fastfm":

    cdef cppclass Settings:
        Settings()
        Settings(string settings)

    cdef cppclass Model:
        Model()
        void add_parameter(double* data, int rank,
                           int n_features, const int order)
        void add_parameter(double* data, int n_features)
        void add_parameter(double* intercept)


    cdef cppclass Data:
        Data()
        void add_design_matrix(int n_samples, int n_features, int nnz,
                               int* outer_ptr, int* inter_ptr, double* data)

        void add_target(const int n_samples, double *data)
        void add_prediction(const int n_samples, double* data)

    cdef void fit(Settings* s, Model* m, Data* d)

    cdef void predict(Model* m, Data* d)
