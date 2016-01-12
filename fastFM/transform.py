# Author: Immanuel Bayer
# License: BSD 3 clause

import scipy.sparse as sp
import numpy as np


def multiclass_to_ranking(X, y):
    n_classes = y.shape[1]
    n_samples = X.shape[0]

    # create extended X matrix
    X_features = X.copy()
    for i in range(n_classes - 1):
        X_features = sp.vstack([X_features, X])

    X_labels = None
    for i in range(n_classes):
        X_tmp = sp.csc_matrix((n_samples, n_classes))
        X_tmp[:, i] = 1
        if X_labels is not None:
            X_labels = sp.vstack([X_labels, X_tmp])
        else:
            X_labels = X_tmp

    X_ext = sp.hstack([X_labels, X_features])

    # create all combinations
    compars = []

    for i_row, row in enumerate(y.tocsr()):
        # over all true labels
        for i in row.indices:
            for c in range(n_classes):
                if c not in row.indices:
                    offset = i_row * n_classes
                    compars.append([offset + i, offset + c])

    compars = np.vstack(compars)
    compars = compars.astype(np.float64)
    return X_ext, compars


def one_hot_encode_df(df, target_col='target', vectorizer=None):
    """ Convert DataFrame to y, X

    Parameters
    ----------
    df: pandas DataFrame

    target_col: str
                String indicating column name of target variable
    vectorizer: sklearn DictVectorizer
                Vectorizer that should be used to encode features

    Returns:
    -------

    X: scipy sparse matrix
        one hot encoded design matrix
    y: array, shape (n_samples)

    vectorizer: sklearn DictVectorizer
                Vectorizer used to encode features

     one-hot encoded scipy sparse matrix
    """

    if target_col not in df.columns:
        raise Exception('target column "' + target_col +
                        '" is not in DataFrame')

    from sklearn.feature_extraction import DictVectorizer
    y = df[target_col].values

    df_dict = df.drop(target_col, axis=1).T.to_dict()
    records = [df_dict[key] for key in df.index.values]
    del df_dict

    if vectorizer is None:
        vectorizer = DictVectorizer(sparse=True)
        vectorizer.fit(records)

    X = vectorizer.transform(records)
    assert len(y) == X.shape[0]

    return X.tocsc(), y.astype(np.float64), vectorizer


def ranking_comparisions(df, group, score):
    """
    Get all pair such that score(i) > score(j), but only compare values
    that belong to the same group

    Parameters
    ----------

    df : DataFrame

i   group: str
        Name of column that's used for the grouping

    score: str
        Name of column that's used for the compares

    Returns
    -------

    comparisions : [(i,j), (, )], list of tuples
        The list of all valid index tuples. The first
        entry is the index of the bigger value.
        score(i) > score(j)
    """
    small = []
    big = []
    from itertools import combinations

    for g, grp in df.groupby(group):
        for i, j in combinations(grp.index, 2):
            # skip draws
            if grp.score[i] > grp.score[j]:
                small.append(j)
                big.append(i)
            elif grp.score[i] < grp.score[j]:
                small.append(i)
                big.append(j)
    return zip(big, small)
