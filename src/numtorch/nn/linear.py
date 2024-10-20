try:
    import cupy as cp
except ImportError:
    import numpy as cp
from numtorch.autograd import Tensor
from typing import Union
from numtorch.nn.base import Module


class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        assert isinstance(in_features, int) and in_features > 0
        assert isinstance(out_features, int) and out_features > 0

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(cp.random.randn(in_features, out_features), requires_grad=True)
        self.bias = Tensor(cp.random.randn(out_features), requires_grad=True)

        self._backward = lambda: None

    def forward(self, x: Tensor):
        out = Tensor(
            cp.matmul(x._data, self.weight._data) + self.bias._data,
            requires_grad=True,
            dtype=x.dtype,
            children=(x, self.weight, self.bias),
            op="nn.Linear",
        )

        def backward():
            # o = x * w + b,
            # do/dw = x
            # do/db = 1
            # do/dx = w
            x.grad += cp.matmul(out.grad, self.weight._data.T)
            self.weight.grad += cp.matmul(x._data.T, out.grad)
            self.bias.grad += out.grad.sum(axis=0)

        out._set_backward(backward)

        return out


if __name__ == "__main__":
    cp.random.seed(0)
    layer = Linear(2, 3)
    layer_weight = cp.random.randn(2, 3)
    layer.weight = Tensor(layer_weight)
    layer.bias = Tensor(cp.ones(3))
    x = Tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = layer(x)
    print(y)
    y.backward()
    print(x.grad)

    try:
        import torch

        x = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.float32, requires_grad=True)
        layer = torch.nn.Linear(2, 3)
        # layer.weight = torch.nn.Parameter(torch.ones((3, 2), dtype=torch.float32, requires_grad=True))
        w = torch.from_numpy(layer_weight)
        w._requires_grad = True
        layer.weight = torch.nn.Parameter(w.T.float())

        layer.bias = torch.nn.Parameter(torch.ones(3, dtype=torch.float32, requires_grad=True))
        y = layer(x)
        print(y)

        y.backward(torch.ones_like(y))
        print(x.grad)
    except ImportError:
        pass
