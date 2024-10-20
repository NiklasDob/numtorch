try:
    import cupy as cp
except ImportError:
    import numpy as cp
from numtorch.autograd import Value, Tensor
from typing import Union


def tanh(v: Union[Value, Tensor]):
    if isinstance(v, Tensor):
        out = Tensor(cp.tanh(v._data), requires_grad=v._requires_grad, dtype=v.dtype, children=(v,), op="tanh")

        def _backward():
            v.grad += (1 - out._data**2) * out.grad

        out._set_backward(_backward)

    else:
        out = Value(cp.tanh(v.value), children=(v,), op="tanh")

        def _backward():
            v.grad += (1 - out.value**2) * out.grad

        out._backward = _backward

    return out


if __name__ == "__main__":
    x = Tensor([1, 2, 3])
    y = tanh(x)
    print(y)
    y.backward()
    print(y.grad)
