

from typing import Tuple, Union
from numtorch.autograd.tensor import Tensor


try:
    import cupy as cp
except ImportError:
    import numpy as cp

import cupy
from cupy._core import internal

# Copied from https://github.com/cupy/cupy/blob/main/cupy/lib/_shape_base.py#L96
# Since put_along_axis is not in the stable release of cupy yet
def _make_along_axis_idx(arr_shape, indices, axis):
    # compute dimensions to iterate over

    if not cupy.issubdtype(indices.dtype, cupy.integer):
        raise IndexError('`indices` must be an integer array')
    if len(arr_shape) != indices.ndim:
        raise ValueError(
            "`indices` and `arr` must have the same number of dimensions")

    shape_ones = (1, ) * indices.ndim
    dest_dims = list(range(axis)) + [None] + \
        list(range(axis + 1, indices.ndim))

    # build a fancy index, consisting of orthogonal cupy.arange calls,
    # with the requested index inserted at the right location
    fancy_index = []
    for dim, n in zip(dest_dims, arr_shape):
        if dim is None:
            fancy_index.append(indices)
        else:
            ind_shape = shape_ones[:dim] + (-1,) + shape_ones[dim+1:]
            fancy_index.append(cupy.arange(n).reshape(ind_shape))

    return tuple(fancy_index)

def put_along_axis(arr, indices, values, axis):
    """
    Put values into the destination array by matching 1d index and data slices.

    This iterates over matching 1d slices oriented along the specified axis in
    the index and data arrays, and uses the former to place values into the
    latter. These slices can be different lengths.

    Functions returning an index along an axis, like `argsort` and
    `argpartition`, produce suitable indices for this function.

    Args:
        arr : cupy.ndarray (Ni..., M, Nk...)
            Destination array.
        indices : cupy.ndarray (Ni..., J, Nk...)
            Indices to change along each 1d slice of `arr`. This must match the
            dimension of arr, but dimensions in Ni and Nj may be 1 to broadcast
            against `arr`.
        values : array_like (Ni..., J, Nk...)
            values to insert at those indices. Its shape and dimension are
            broadcast to match that of `indices`.
        axis : int
            The axis to take 1d slices along. If axis is None, the destination
            array is treated as if a flattened 1d view had been created of it.

    .. seealso:: :func:`numpy.put_along_axis`
    """

    # normalize inputs
    if axis is None:
        if indices.ndim != 1:
            raise NotImplementedError(
                "Tuple setitem isn't supported for flatiter.")
        # put is roughly equivalent to a.flat[ind] = values
        cupy.put(arr, indices, values)
    else:
        axis = internal._normalize_axis_index(axis, arr.ndim)
        arr_shape = arr.shape

        # use the fancy index
        arr[_make_along_axis_idx(arr_shape, indices, axis)] = values


def max(x: Tensor, axis: Union[int, Tuple[int, ...], None] = None, keepdims:bool=False):
    max_ind = cp.argmax(x._data, axis=axis, keepdims=True)

    # NOTE: This does not work with tuple axis
    # out_data = cp.take_along_axis(x._data, max_ind, axis=axis)

    out = Tensor(cp.max(x._data, axis=axis, keepdims=keepdims), requires_grad=x._requires_grad, dtype=x.dtype, children=(x,), op="nt.max")

    mask = cp.zeros_like(x._data, dtype=bool)

    if axis is None or isinstance(axis, int):
        put_along_axis(mask, max_ind, True, axis=axis)
    else:
        #NOTE: This is a very hacky way of doing this, but i have found no other way to do it
        # put_along_axis does not support tuple axis
        # So if you find some better way of doing this please let me know

        flatten_axes = sorted([ax if ax >= 0 else ax + x._data.ndim for ax in axis])

        # Reshape the array to combine the axes
        new_shape = list(x._data.shape)
        flattened_size = 1
        for ax in flatten_axes:
            flattened_size *= new_shape[ax]

        # Reshape to combine the axes
        new_shape[flatten_axes[0]] = flattened_size
        for ax in sorted(flatten_axes[1:], reverse=True):
            new_shape.pop(ax)

        reshaped_data = x._data.reshape(new_shape)

        # Apply the same process as for a single axis
        max_ind_flattened = cp.argmax(reshaped_data, axis=flatten_axes[0], keepdims=True)
        mask_flattened = cp.zeros_like(reshaped_data, dtype=bool)
        put_along_axis(mask_flattened, max_ind_flattened, True, axis=flatten_axes[0])

        # Reshape the mask back to the original dimensions
        mask = mask_flattened.reshape(x.shape)



    def backward():
        if x._requires_grad:
            x.grad[mask] += out.grad.reshape(x.grad[mask].shape)

    out._set_backward(backward)

    return out

if __name__ == "__main__":
    # Example testing of max function
    x = Tensor(cp.array([[1.0, 2.0, 3.0], [4.0, 7.0, 6.0]]), requires_grad=True)
    y = max(x, axis=1)
    print(f"Max result: {y._data}")
    y.backward()
    print(f"Gradient wrt x: {x.grad}")

    x = Tensor(cp.random.random((2,1,10,10)), requires_grad=True)
    y = max(x, axis=(2,3))
    print(f"Max result: {y._data}")
    y.backward()
    print(f"Gradient wrt x: {x.grad}")

