Tutorials
=========

The following sections show how to use different features of the fastFM
library. This is mostly a demonstration on of the library and no background
on the Factorization Machine (FM) model is given.
I recommend to read [TIST2012]. This paper contains many examples on how FM's
can emulate and extend matrix factorization models through feature engineering.


Regression with ALS Solver
--------------------------

We first set up a small toy dataset for a regression problem. Please
refere to [SIGIR2011] for background information on the implemented ALS solver.

.. testcode::

    from fastFM.datasets import make_user_item_regression
    from sklearn.cross_validation import train_test_split

    # This sets up a small test dataset.
    X, y, _ = make_user_item_regression(label_stdev=.4)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

The number of iterations `n_iter`, the standard deviation `init_stdev` used to
initialize the model parameter and the number of hidden variables `rank` per feature.
This are the parameters that have to be specified for every solver and task. The ALS
solver requires in addition the regularization values for the first `l2_reg_w`
and second order `l2_reg_V` interactions.

.. testcode::

    from fastFM import als
    fm = als.FMRegression(n_iter=1000, init_stdev=0.1, rank=2, l2_reg_w=0.1, l2_reg_V=0.5)
    fm.fit(X_train, y_train)
    y_pred = fm.predict(X_test)

We can easily evaluate our model using the scikit-learn library.

.. testcode::

    from sklearn.metrics import mean_squared_error
    'mse:', mean_squared_error(y_test, y_pred)


Logit Classification with SGD Solver
------------------------------------

We first have to convert the target of our toy dataset to -1/1 values
in order to work with the classification implementation. Currently only
binary classification is supported.

.. testcode::

    import numpy as np
    # Convert dataset to binary classification task.
    y_labels = np.ones_like(y)
    y_labels[y < np.mean(y)] = -1
    X_train, X_test, y_train, y_test = train_test_split(X, y_labels)


We could have used the ALS solver module for this problem as well but
we will use the SGD module instead. In addition to the
hyper parameter needed for the ALS module we need to specify
the SGD specific `step_size` parameter.

.. testcode::

    from fastFM import sgd
    fm = sgd.FMClassification(n_iter=1000, init_stdev=0.1, l2_reg_w=0,
                              l2_reg_V=0, rank=2, step_size=0.1)
    fm.fit(X_train, y_train)
    y_pred = fm.predict(X_test)


All classifier implementations can not only return the most likely labels
but also class probabilities via the `predict_proba`.

.. testcode::

    y_pred_proba = fm.predict_proba(X_test)

This is important for classification metrics such as the AUC score that require the class probabilities
as input.

.. testcode::

    from sklearn.metrics import accuracy_score, roc_auc_score
    'acc:', accuracy_score(y_test, y_pred)
    'auc:', roc_auc_score(y_test, y_pred_proba)


Bayesian Probit Classification with MCMC Solver
-----------------------------------------------

The MCMC module needs fewer hyper parameter that any other solver.
This solver is able to integrate out the regularization parameter and frees us
from selecting them manually. Please see [Freuden2011] for the detail on the implemented
Gibbs sampler.
The major drawback of the MCMC solver is that it forces us to calculate predictions
during fitting time using the `fit_predict` function.
It's however possible to select a subset of parameter draws to speed up prediction [RecSys2013].
It's also possible to just call `predict` on a trained MCMC model but this returns predictions
that are solely based on the last parameters draw.
These predictions can be used for diagnostic purposes but
are usually not as good as averaged predictions returned by `fit_predict`.


.. testcode::

    from fastFM import mcmc
    fm = mcmc.FMClassification(n_iter=1000, rank=2, init_stdev=0.1)

Our last example shows how to use the MCMC module for binary classification.
Probit regression uses the Cumulative Distribution Function (CDF) of the standard normal Distribution
as link function. Mainly because the CDF leads to an easier Gibbs solver then the
sigmoid function used in the SGD classifier implementation. The results
are in practice usually very similar.

.. testcode::

    y_pred = fm.fit_predict(X_train, y_train, X_test)
    y_pred_proba = fm.fit_predict_proba(X_train, y_train, X_test)


.. testcode::

    from sklearn.metrics import accuracy_score, roc_auc_score
    'acc:', accuracy_score(y_test, y_pred)
    'auc:', roc_auc_score(y_test, y_pred_proba)



.. [TIST2012] Rendle, Steffen. "Factorization machines with libfm." ACM Transactions on Intelligent Systems and Technology (TIST) 3.3 (2012): 57.
.. [SIGIR2011] Rendle, Steffen, et al. "Fast context-aware recommendations with factorization machines." Proceedings of the 34th international ACM SIGIR conference on Research and development in Information Retrieval. ACM, 2011.
.. [Freuden2011] C Freudenthaler, L Schmidt-Thieme, S Rendle "Bayesian factorization machines" - 2011 - Citeseer
.. [RecSys2013] Silbermann, Bayer, and Rendle "Sample selection for MCMC-based recommender systems" Proceedings of the 7th ACM conference on Recommender systems 2013
