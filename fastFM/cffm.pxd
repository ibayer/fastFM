#cfastfm.pxd
#
# Declarations of "external" C functions and structures
# distutils: include_dirs = /usr/include/suitesparse

cdef extern from "../fastFM-core/externals/CXSparse/Include/cs.h":
    ctypedef struct cs_di:  # matrix in compressed-column or triplet form */
        int nzmax      # maximum number of entries */
        int m          # number of rows */
        int n          # number of columns */
        int *p         # column pointers (size n+1) or col indices (size nzmax) */
        int *i         # row indices, size nzmax */
        double *x      # numerical values, size nzmax */
        int nz         # # of entries in triplet matrix, -1 for compressed-col */

    double cs_di_norm(const cs_di *X) # max colsum

cdef extern from "../fastFM-core/include/ffm.h":

    ctypedef struct ffm_param:
        int n_iter
        int k
        double init_sigma
        double lambda_w
        double lambda_V
        int TASK
        double stepsize
        int rng_seed

        int ignore_w_0
        int ignore_w
        int keep_coef

        int n_hyper_param
        double *hyper_param

    void ffm_predict(double *w_0, double * w, double * V, cs_di *X, double *y_pred, int k)

    void ffm_als_fit(double *w_0, double *w, double *V,
        cs_di *X, double *y, ffm_param *param)

    void ffm_mcmc_fit_predict(double *w_0, double *w, double *V,
        cs_di *X_train, cs_di *X_test, double *y_train, double *y_pred,
        ffm_param *param)

    void ffm_sgd_fit(double *w_0, double *w, double *V,
        cs_di *X, double *y, ffm_param *param)

    void ffm_sgd_bpr_fit(double *w_0, double *w, double *V,
        cs_di *X, double *pairs, int n_pairs, ffm_param *param)
