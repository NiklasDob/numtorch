from __future__ import annotations
from typing import Tuple, Union
from numtorch.autograd.value import Value

try:
    import cupy as cp
except ImportError:
    print("Could not import cupy falling back to numpy...")
    import numpy as cp


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
        if dim1 != dim2 and dim2 != 1 and dim1 != 1:
            raise ValueError(f"Cannot broadcast: dimension mismatch between t1 {shape1} and t2 {shape2}")

    expanded_data = cp.broadcast_to(t2._data, shape1)
    old_shape = t2._data.shape
    new_shape = expanded_data.shape
    out = Tensor(expanded_data, children=(t2,), op="broadcast", requires_grad=t1._requires_grad or t2._requires_grad)

    def _backward():
        """
        Backward function for broadcasting. The gradient is summed along the dimensions where t2 was broadcasted.
        """
        grad = out.grad

        reduced_grad = grad
        rev_shape1 = shape1[::-1]
        rev_shape2 = shape2[::-1]
        sum_axis = []
        for i in range(len(rev_shape1)):
            dim1 = rev_shape1[i]
            if i < len(rev_shape2):
                dim2 = rev_shape2[i]

                if dim1 != dim2:
                    sum_axis.append(len(reduced_grad.shape) - 1 - i)
            else:
                sum_axis.append(len(reduced_grad.shape) - 1 - i)


        reduced_grad = cp.sum(reduced_grad, axis=sum_axis, keepdims=True)

        reduced_grad = cp.squeeze(reduced_grad)

        t2.grad += reduced_grad

    out._set_backward(_backward)

    return out


class Tensor:
    def __init__(
        self,
        arr,
        children: Tuple[Tensor, ...] = (),
        op: Union[str, None] = None,
        dtype=cp.float32,
        requires_grad: bool = False,
    ):
        assert (
            isinstance(arr, list)
            or isinstance(arr, tuple)
            or isinstance(arr, cp.ndarray)
            or isinstance(arr, float)
            or isinstance(arr, int)
        )
        # data = convert_array_to_value_arr(arr)
        data = cp.array(arr, dtype=dtype)
        self.dtype = dtype
        self._requires_grad = requires_grad
        self._data = data

        self.grad: cp.ndarray = cp.zeros_like(data) if requires_grad else None

        self._backward = []
        self._children = children
        self._op = op

    def internal_backward(self):
        for func in reversed(self._backward):
            func()

    def _set_backward(self, func):
        if self._requires_grad:
            self._backward.append(func)

    def item(self):
        return self._data.item()

    def __getitem__(self, index):
        out = Tensor(
            self._data[index], requires_grad=self._requires_grad, dtype=self.dtype, children=(self,), op="__getitem__"
        )

        def _backward():
            self.grad[index] += out.grad

        out._set_backward(_backward)
        return out

    @property
    def shape(self):
        def get_shape(data):
            if isinstance(data, list):
                return (len(data),) + get_shape(data[0])
            else:
                return data.shape

        return get_shape(self._data)

    def __setitem__(self, index, value):
        """
        Allows in-place modification of tensor values at specific indices.
        For example: x[:, 0] = 1
        """
        if isinstance(value, Tensor):
            self._data[index] = value._data

            self._children += (value,)

            def _backward():
                value.grad += self.grad[index]

            value._set_backward(_backward)
            # # Set the backward method for this modification
            # self._set_backward(_backward)
        else:
            # Handle non-tensor values (e.g., setting with scalars or arrays)
            self._data[index] = value

    def reshape(self, *shape) -> Tensor:
        original_shape = self._data.shape
        out = Tensor(
            self._data.reshape(*shape),
            requires_grad=self._requires_grad,
            dtype=self.dtype,
            children=(self,),
            op="reshape",
        )

        def _backward():
            self.grad += out.grad.reshape(original_shape)

        out._set_backward(_backward)
        return out

    def __len__(self):
        return len(self._data)

    def _convert_other(self, other):
        out = other if isinstance(other, Tensor) else Tensor(other)
        if self.shape != out.shape and len(self.shape) != len(out.shape):
            out = broadcast_to(self, out)
        return out

    def __add__(self, other):
        other = self._convert_other(other)
        out = Tensor(
            self._data + other._data,
            requires_grad=self._requires_grad or other._requires_grad,
            dtype=self.dtype,
            children=(self, other),
            op="+",
        )

        def _backward():
            if self._requires_grad:
                if self.shape == out.shape:
                    self.grad += out.grad
                else:
                    sum_axis = []
                    for i in range(len(out.shape)):
                        if out.shape[i] != self.shape[i]:
                            sum_axis.append(i)

                    self.grad += cp.sum(out.grad, axis=sum_axis, keepdims=True)

            if other._requires_grad:
                if other.shape == out.shape:
                    other.grad += out.grad
                else:
                    sum_axis = []
                    for i in range(len(out.shape)):
                        if out.shape[i] != other.shape[i]:
                            sum_axis.append(i)

                    other.grad += cp.sum(out.grad, axis=sum_axis, keepdims=True)

        out._set_backward(_backward)

        return out

    def __mul__(self, other):
        other = self._convert_other(other)
        out = Tensor(
            self._data * other._data,
            requires_grad=self._requires_grad or other._requires_grad,
            dtype=self.dtype,
            children=(self, other),
            op="mul",
        )

        def _backward():
            if self._requires_grad:
                self_up = other._data * out.grad
                if self.shape == self_up.shape :
                    self.grad += self_up
                else:
                    sum_axis = []
                    for i in range(len(self_up.shape)):
                        if self_up.shape[i] != self.shape[i]:
                            sum_axis.append(i)

                    self.grad += cp.sum(self_up, axis=sum_axis, keepdims=True)

            if other._requires_grad:
                other_up = self._data * out.grad
                if other.shape == other_up.shape :
                    other.grad += other_up
                else:
                    sum_axis = []
                    for i in range(len(other_up.shape)):
                        if other_up.shape[i] != other.shape[i]:
                            sum_axis.append(i)

                    other.grad += cp.sum(other_up, axis=sum_axis, keepdims=True)

        out._set_backward(_backward)

        return out

    def __pow__(self, other):
        other = self._convert_other(other)
        out = Tensor(
            self._data**other._data,
            requires_grad=self._requires_grad or other._requires_grad,
            dtype=self.dtype,
            children=(self, other),
            op="pow",
        )

        def _backward():
            if self._requires_grad:
                self.grad += (other._data * self._data ** (other._data - 1)) * out.grad
            if other._requires_grad:
                other.grad += out._data * cp.log(self._data) * out.grad

        out._set_backward(_backward)

        return out

    def __neg__(self):
        out = self * Tensor(-1, requires_grad=self._requires_grad, dtype=self.dtype, children=(self,), op="-")
        return out

    def __radd__(self, other):  # other + self
        o = other._convert_other(self)
        return o + other

    def __rsub__(self, other):  # other - self
        o = other._convert_other(self)
        return other + (-o)

    def __rmul__(self, other):  # other * self
        o = other._convert_other(self)
        return o * other

    def __truediv__(self, other):  # self / other

        return self * other ** (-1)

    def __rtruediv__(self, other):  # other / self
        o = other._convert_other(self)

        return other * o ** (-1)

    def backward(self):
        self.grad = cp.ones_like(self._data)

        topo = []
        seen = set()

        def build_topo_sort(child):
            if child in seen:
                return
            seen.add(child)
            for c in child._children:
                build_topo_sort(c)
            topo.append(child)

        build_topo_sort(self)

        for node in reversed(topo):
            node.internal_backward()

    def __repr__(self) -> str:
        return f"Tensor({self._data})"


if __name__ == "__main__":
    # https://cupy.dev/
    # micromamba install -c conda-forge cupy-core
    # only numpy
    x = Tensor([[1], [2], [3]], requires_grad=False)
    y = Tensor([[2], [2], [2]], requires_grad=True)
    # TODO: Test if the gradient is correct
    z = Tensor(1)

    z = x**y
    # z = a ** y
    # dz / dy = a**y * log(a)
    # z = -z
    # z = z / 2
    z.backward()

    print("x.grad:", x.grad)
    print("y.grad:", y.grad)
