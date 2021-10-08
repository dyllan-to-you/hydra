import numpy as np
import numba as nb


@nb.njit(parallel=True)
def in1d_vec_nb(matrix, index_to_remove, invert):
    # matrix and index_to_remove have to be numpy arrays
    # if index_to_remove is a list with different dtypes this
    # function will fail

    out = np.empty(matrix.shape[0], dtype=nb.boolean)
    index_to_remove_set = set(index_to_remove)

    for i in nb.prange(matrix.shape[0]):
        if matrix[i] in index_to_remove_set:
            out[i] = invert
        else:
            out[i] = not invert

    return out


@nb.njit(parallel=True)
def in1d_scal_nb(matrix, index_to_remove, invert):
    # matrix and index_to_remove have to be numpy arrays
    # if index_to_remove is a list with different dtypes this
    # function will fail

    out = np.empty(matrix.shape[0], dtype=nb.boolean)
    for i in nb.prange(matrix.shape[0]):
        if matrix[i] == index_to_remove:
            out[i] = invert
        else:
            out[i] = not invert

    return out


def isin_nb(matrix_in, index_to_remove, invert=False):
    # both matrix_in and index_to_remove have to be a np.ndarray
    # even if index_to_remove is actually a single number
    shape = matrix_in.shape
    if index_to_remove.shape == ():
        res = in1d_scal_nb(matrix_in.reshape(-1), index_to_remove.take(0), invert)
    else:
        res = in1d_vec_nb(matrix_in.reshape(-1), index_to_remove, invert)

    return res.reshape(shape)
