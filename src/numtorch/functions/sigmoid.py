try:
    import cupy as cp
except ImportError:
    import numpy as cp

from numtorch.autograd import Tensor, Value
from typing import Union


def sigmoid(v: Union[Value, Tensor]):
    if isinstance(v, Tensor):
        out_data = 1 / (1 + cp.exp(-v._data))
        out = Tensor(
            out_data,
            children=(v,),
            op="sigmoid",
            requires_grad=v._requires_grad,
            dtype=v.dtype,
        )

        def _backward():
            # Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
            sigmoid_grad = out._data * (1 - out._data)
            v.grad += sigmoid_grad * out.grad

        out._set_backward(_backward)

    else:
        out_value = 1 / (1 + cp.exp(-v.value))
        out = Value(
            out_value,
            children=(v,),
            op="sigmoid",
        )

        def _backward():
            # Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
            sigmoid_grad = out.value * (1 - out.value)
            v.grad += sigmoid_grad * out.grad

        out._backward = _backward

    return out


if __name__ == "__main__":
    # Testing with a Value instance
    x = Value(2)
    y = sigmoid(x)
    print(y)
    y.backward()
    print("Gradient of x:", x.grad)

    # Testing with a Tensor instance
    x = Tensor([[1], [2], [3]])
    y = sigmoid(x)
    print(y)
    y.backward()
    print("Gradient of Tensor x:\n", x.grad)
