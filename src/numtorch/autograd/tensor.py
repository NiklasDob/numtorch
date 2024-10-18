from numtorch.autograd.value import Value
import numpy as np


def convert_array_to_value_arr(arr):
    """
    Converts each element of the array to a Value object. This is also for multi-dimensional arrays the shape of the array is [n, m, ...]
    """

    def check(list):
        return all(i == list[0] for i in list)

    assert check([len(e) for e in arr if isinstance(e, list)]), "All elements in the array must have the same length"

    if len(arr.shape) == 0:
        return [Value(arr)]
    else:
        return [convert_array_to_value_arr(arr[i]) for i in range(len(arr))]


def broadcast_to(t1: Tensor, t2: Tensor):
    """
    Broadcasts t2 to the shape of t1 if possible.
    Raises a ValueError if broadcasting is not possible.
    """
    shape1 = t1.shape
    shape2 = t2.shape

    # Check if broadcasting is possible
    if len(shape2) > len(shape1):
        raise ValueError(f"Cannot broadcast: shape of t2 {shape2} is larger than shape of t1 {shape1}")

    # Add leading dimensions to shape2 if necessary
    shape2 = (1,) * (len(shape1) - len(shape2)) + shape2

    # Verify that broadcasting is possible for each dimension
    for dim1, dim2 in zip(shape1, shape2):
        if dim1 != dim2 and dim2 != 1:
            raise ValueError(f"Cannot broadcast: dimension mismatch between t1 {shape1} and t2 {shape2}")

    # Create a new Tensor with the broadcasted values
    def expand(data, target_shape):
        """
        Recursively expand the dimensions of data to match the target shape.
        """
        if isinstance(data, list):
            # If we're already at the correct shape, return data
            if len(data) == target_shape[0]:
                return [expand(d, target_shape[1:]) for d in data]
            # If the current dimension is 1, broadcast it by repeating the data
            elif len(data) == 1:
                return [expand(data[0], target_shape[1:]) for _ in range(target_shape[0])]
            else:
                raise ValueError("Data cannot be expanded to the target shape")
        else:
            # This is a Value object; no further expansion needed
            return data

    expanded_data = expand(t2._data, shape1)
    out = Tensor(expanded_data, children=(t2,), op="broadcast")

    def _backward():
        """
        Backward function for broadcasting. The gradient is summed along the dimensions where t2 was broadcasted.
        """
        grad = out.grad

        reduced_grad = grad
        for i, (dim1, dim2) in enumerate(zip(shape1, shape2)):
            if dim2 == 1:
                reduced_grad = np.sum(reduced_grad, axis=i, keepdims=True)

        reduced_grad = np.squeeze(reduced_grad)

        t2.grad += reduced_grad

    out._backward = _backward

    return out


class Tensor:
    def __init__(self, arr, children=(), op=None):
        assert isinstance(arr, list) or isinstance(arr, tuple) or isinstance(arr, np.ndarray)
        data = convert_array_to_value_arr(arr)
        self._data = data
        self.grad = np.zeros_like(data)

        self._backward = lambda: None
        self.children = children
        self.op = op

    @property
    def shape(self):
        def get_shape(data):
            if isinstance(data, list):
                return (len(data),) + get_shape(data[0])
            else:
                return data.shape

        return get_shape(self._data)

    def __len__(self):
        return len(self._data)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self.shape != other.shape:
            other = broadcast_to(self, other)

        out = Tensor(self._data + other._data, children=(self, other), op="+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __repr__(self) -> str:
        return f"Tensor({self._data})"
