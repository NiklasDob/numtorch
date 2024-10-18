import numpy as np
from numtorch.autograd import Value


def tanh(v: Value):
    out = Value(np.tanh(v.value), children=(v,), op="tanh")

    def _backward():
        v.grad += (1 - out.value**2) * out.grad

    out._backward = _backward
    return out


if __name__ == "__main__":
    x = Value(1)
    y = tanh(x)
    print(y)
    y.backward()
    print(y.grad)
