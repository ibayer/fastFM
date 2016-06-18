# Static versions of non-core sklearn.utils functions.
# Placed here since they are subject to change.

"""Utilities for input validation"""

# Authors: Olivier Grisel
#          Gael Varoquaux
#          Andreas Mueller
#          Lars Buitinck
#          Alexandre Gramfort
#          Nicolas Tresegnie
# License: BSD 3 clause

import numbers
import warnings

import numpy as np
import scipy.sparse as sparse
from functools import wraps


def _check_matrix_is_sparse(func):
    """
    Check that input is a scipy sparse matrix and raise warning otherwise.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'accept_sparse' in kwargs and not sparse.isspmatrix(args[0]):
            raise TypeError('A dense matrix was passed in, but sparse'
                            'data is required.')
        result = func(*args, **kwargs)
        return result
    return wrapper


def _ensure_sparse_format(spmatrix, accept_sparse, dtype, order, copy,
                          force_all_finite):
    """Convert a sparse matrix to a given format.
    Checks the sparse format of spmatrix and converts if necessary.
    Parameters
    ----------
    spmatrix : scipy sparse matrix
        Input to validate and convert.
    accept_sparse : string, list of string or None (default=None)
        String[s] representing allowed sparse matrix formats ('csc',
        'csr', 'coo', 'dok', 'bsr', 'lil', 'dia'). None means that sparse
        matrix input will raise an error.  If the input is sparse but not in
        the allowed format, it will be converted to the first listed format.
    dtype : string, type or None (default=none)
        Data type of result. If None, the dtype of the input is preserved.
    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.
    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.
    force_all_finite : boolean (default=True)
        Whether to raise an error on np.inf and np.nan in X.
    Returns
    -------
    spmatrix_converted : scipy sparse matrix.
        Matrix that is ensured to have an allowed type.
    """
    if accept_sparse is None:
        raise TypeError('A sparse matrix was passed, but dense '
                        'data is required. Use X.toarray() to '
                        'convert to a dense numpy array.')
    sparse_type = spmatrix.format
    if dtype is None:
        dtype = spmatrix.dtype
    if sparse_type in accept_sparse:
        # correct type
        if dtype == spmatrix.dtype:
            # correct dtype
            if copy:
                spmatrix = spmatrix.copy()
        else:
            # convert dtype
            spmatrix = spmatrix.astype(dtype)
    else:
        # create new
        spmatrix = spmatrix.asformat(accept_sparse[0]).astype(dtype)
    if force_all_finite:
        if not hasattr(spmatrix, "data"):
            warnings.warn("Can't check %s sparse matrix for nan or inf."
                          % spmatrix.format)
        else:
            assert_all_finite(spmatrix.data)
    if hasattr(spmatrix, "data"):
        spmatrix.data = np.array(spmatrix.data, copy=False, order=order)
    return spmatrix


def assert_all_finite(X):
    """Like assert_all_finite, but only for ndarray."""
    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method.
    if (X.dtype.char in np.typecodes['AllFloat'] and
            not np.isfinite(X.sum()) and not np.isfinite(X).all()):
        raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r." % X.dtype)


@_check_matrix_is_sparse
def check_array(array, accept_sparse=None, dtype="numeric", order=None,
                copy=False, force_all_finite=True, ensure_2d=True,
                allow_nd=False, ensure_min_samples=1, ensure_min_features=1):
    """Input validation on an array, list, sparse matrix or similar.
    By default, the input is converted to an at least 2nd numpy array.
    If the dtype of the array is object, attempt converting to float,
    raising on failure.
    Parameters
    ----------
    array : object
        Input object to check / convert.
    accept_sparse : string, list of string or None (default=None)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc.  None means that sparse matrix input will raise an error.
        If the input is sparse but not in the allowed format, it will be
        converted to the first listed format.
    dtype : string, type or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.
    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.
    force_all_finite : boolean (default=True)
        Whether to raise an error on np.inf and np.nan in X.
    ensure_2d : boolean (default=True)
        Whether to make X at least 2d.
    allow_nd : boolean (default=False)
        Whether to allow X.ndim > 2.
    ensure_min_samples : int (default=1)
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.
    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.
    Returns
    -------
    X_converted : object
        The converted and validated X.
    """
    if isinstance(accept_sparse, str):
        accept_sparse = [accept_sparse]

    # store whether originally we wanted numeric dtype
    dtype_numeric = dtype == "numeric"

    if sparse.issparse(array):
        if dtype_numeric:
            dtype = None
        array = _ensure_sparse_format(array, accept_sparse, dtype, order,
                                      copy, force_all_finite)
    else:
        if ensure_2d:
            array = np.atleast_2d(array)
        if dtype_numeric:
            if (hasattr(array, "dtype") and
                    getattr(array.dtype, "kind", None) == "O"):
                # if input is object, convert to float.
                dtype = np.float64
            else:
                dtype = None
        array = np.array(array, dtype=dtype, order=order, copy=copy)
        # make sure we actually converted to numeric:
        if dtype_numeric and array.dtype.kind == "O":
            array = array.astype(np.float64)
        if not allow_nd and array.ndim >= 3:
            raise ValueError("Found array with dim %d. Expected <= 2" %
                             array.ndim)
        if force_all_finite:
            assert_all_finite(array)

    shape_repr = _shape_repr(array.shape)
    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError("Found array with %d sample(s) (shape=%s) while a"
                             " minimum of %d is required."
                             % (n_samples, shape_repr, ensure_min_samples))

    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError("Found array with %d feature(s) (shape=%s) while"
                             " a minimum of %d is required."
                             % (n_features, shape_repr, ensure_min_features))
    return array


def check_consistent_length(x1, x2):
    return x1.shape[0] == x2.shape[0]


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def _shape_repr(shape):
    """Return a platform independent reprensentation of an array shape
    Under Python 2, the `long` type introduces an 'L' suffix when using the
    default %r format for tuples of integers (typically used to store the shape
    of an array).
    Under Windows 64 bit (and Python 2), the `long` type is used by default
    in numpy shapes even when the integer dimensions are well below 32 bit.
    The platform specific type causes string messages or doctests to change
    from one platform to another which is not desirable.
    Under Python 3, there is no more `long` type so the `L` suffix is never
    introduced in string representation.
    >>> _shape_repr((1, 2))
    '(1, 2)'
    >>> one = 2 ** 64 / 2 ** 64  # force an upcast to `long` under Python 2
    >>> _shape_repr((one, 2 * one))
    '(1, 2)'
    >>> _shape_repr((1,))
    '(1,)'
    >>> _shape_repr(())
    '()'
    """
    if len(shape) == 0:
        return "()"
    joined = ", ".join("%d" % e for e in shape)
    if len(shape) == 1:
        # special notation for singleton tuples
        joined += ','
    return "(%s)" % joined


def _num_samples(x):
    """Return number of samples in array-like x."""
    if hasattr(x, 'fit'):
        # Don't get num_samples from an ensembles length!
        raise TypeError('Expected sequence or array-like, got '
                        'estimator %s' % x)
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %s" %
                            type(x))
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        return x.shape[0]
    else:
        return len(x)
