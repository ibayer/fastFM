import scipy.sparse as sp
import numpy as np
from numpy.testing import assert_array_equal
from pandas import DataFrame
from sklearn.datasets import make_classification

from fastFM.bpr import FMRecommender
from fastFM.transform import multiclass_to_ranking, one_hot_encode_df, \
                        ranking_comparisions


def test_one_hot_encode_df():
    df = DataFrame({'user_id':['u1', 'u2','u1', 'u2', 'u3'],
                    'item_id':['i1', 'i2', 'i4', 'i1', 'i2'],
                    'target':[4, 5, 10, 1, 2],
                    }, index=['a', 'd', 'b', 'c', 'f'])
    y_org = df.target.values
    X, y, vec = one_hot_encode_df(df)
    # test one hot encoding
    assert X.shape[1] == 6
    # use the target to check if order is preserved
    assert_array_equal(y_org, df.target.values) # test for side effects
    assert_array_equal(y_org, y)


def test_multiclass_to_ranking():
    X = sp.csr_matrix(np.arange(6).reshape((3,2)))
    y = sp.csr_matrix((3, 5))
    y[0, 0] = 1
    y[1, [2, 3]] = 1
    y[2, [0, 4]] = 1

    n_classes = y.shape[1]
    n_samples = X.shape[0]
    n_features = X.shape[1]

    X_ext, compars = multiclass_to_ranking(X, y)


    assert X_ext.shape[0] == n_classes * n_samples
    assert X_ext.shape[1] == n_classes + n_features

    # test that features are replicated
    assert_array_equal(X_ext.tocsc()[:, n_classes:].sum(axis=0),
            X.sum(axis=0) * n_classes)

    # test class labels encoding structure
    assert X_ext.tocsr()[:n_classes, :].sum(axis=0)[0, 0] == n_samples
    assert X_ext.tocsr()[:, :n_classes].sum() == n_samples * n_classes
    #assert_array_equal(X_ext.tocsc()[:, n_classes:].sum(axis=0),

    print X_ext.todense()
    print y.todense()
    print compars


def test_multiclass_encoding():
    #X, y = make_classification(n_samples=100, n_features=20,
    n_samples =  10
    n_classes = 3

    X, y = make_classification(n_samples=n_samples, n_features=10,
            n_informative=6, n_redundant=2, n_repeated=0,
            n_classes=n_classes, n_clusters_per_class=2,
            weights=None, flip_y=0.01, class_sep=1.0,
            hypercube=True, shift=0.0, scale=1.0,
            shuffle=True, random_state=None)

    rows = np.arange(len(y))
    cols = y
    values = np.ones_like(y, dtype=np.float64)

    y_one_hot = sp.coo_matrix((values, (rows, cols)),
            shape=(n_samples, n_classes), dtype=np.float64)
    y_one_hot = y_one_hot.tocsr()
    X = sp.csr_matrix(X)

    X_ext, compars = multiclass_to_ranking(X, y_one_hot)

    fm = FMRecommender(n_iter=2000,
            init_stdev=0.01, l2_reg_w=.5, l2_reg_V=.5, rank=2,
            step_size=.002, random_state=11)
    X_train = X_ext.tocsc()
    #assert False
    fm.fit(X_train, compars)
    #y_pred = fm.predict(X_test)


def test_ranking_comparisions():
    df = DataFrame({'group':['a', 'a', 'b', 'd', 'a', 'b'],
                    'score':[3, 3, 10, 4, 2, 55]
        })
    print df
    compars = ranking_comparisions(df, 'group', 'score')

    true_comparisions = [(5, 2),

                         (0, 4),
                         (1, 4)]
    assert len(true_comparisions) == len(compars)
    for pair in compars:
        print pair
        assert pair in true_comparisions


if __name__ == '__main__':
    #test_multiclass_to_ranking()
    test_ranking_comparisions()

