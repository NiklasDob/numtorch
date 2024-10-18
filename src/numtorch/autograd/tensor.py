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
    Broadcasts t2 to the shape of t1
    """


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
