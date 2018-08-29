# Author: Immanuel Bayer
# License: BSD 3 clause

cdef extern from "../fastFM-core/externals/CXSparse/Include/cs.h":
    ctypedef struct cs_di:  # matrix in compressed-column or triplet form */
        int nzmax      # maximum number of entries */
        int m          # number of rows */
        int n          # number of columns */
        int *p         # column pointers (size n+1) or col indices (size nzmax) */
        int *i         # row indices, size nzmax */
        double *x      # numerical values, size nzmax */
        int nz         # # of entries in triplet matrix, -1 for compressed-col */

cdef extern from "../fastFM-core/include/ffm.h":

    ctypedef struct ffm_param:
        int n_iter
        int k
        double init_sigma
        double init_lambda_w
        double init_lambda_V
        int TASK
        double stepsize
        int rng_seed

        int iter_count
        int ignore_w_0
        int ignore_w
        int warm_start

        int n_hyper_param
        double *hyper_param

    void ffm_predict(double *w_0, double * w, double * V, cs_di *X, double *y_pred, int k)

    void ffm_als_fit(double *w_0, double *w, double *V,
        cs_di *X, double *y, ffm_param *param)

    void ffm_mcmc_fit_predict(double *w_0, double *w, double *V,
        cs_di *X_train, cs_di *X_test, double *y_train, double *y_pred,
        ffm_param *param)

    void ffm_sgd_bpr_fit(double *w_0, double *w, double *V,
        cs_di *X, double *pairs, int n_pairs, ffm_param *param)
