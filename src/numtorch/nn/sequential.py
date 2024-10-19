try:
    import cupy as cp
except ImportError:
    import numpy as cp
from numtorch.autograd import Tensor
from typing import Union
from numtorch.nn.base import Module


class Sequential(Module):
    def __init__(self, *layers: Module):
        super(Sequential, self).__init__()
        assert all([isinstance(layer, Module) for layer in layers])
        self.layers = layers

    def forward(self, x: Tensor):
        for layer in self.layers:
            x = layer(x)
        return x
