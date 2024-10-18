import numpy as np
from numtorch.autograd import Value


def relu(v: Value):
    out = Value(np.maximum(0, v.value), children=(v,), op="relu")

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
