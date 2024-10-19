try:
    import cupy as cp
except ImportError:
    print("Could not import cupy falling back to numpy...")
    import numpy as cp

from numtorch.autograd import Tensor, Value
from typing import Union


def relu(v: Union[Value, Tensor]):
    if isinstance(v, Tensor):
        out = Tensor(cp.maximum(0, v._data), children=(v,), op="relu")

        def _backward():
            v.grad += (out._data > 0.0).astype(out.grad.dtype) * out.grad

    else:
        out = Value(cp.maximum(0, v.value), children=(v,), op="relu")

        def _backward():
            v.grad += (1.0 if out.value > 0.0 else 0.0) * out.grad

    out._backward = _backward
    return out


if __name__ == "__main__":
    x = Value(2)
    y = relu(x)
    print(y)
    y.backward()
    print(x.grad)

    x = Tensor([[1], [2], [3]])
    y = relu(x)
    print(y)
    y.backward()
    print(x.grad)
