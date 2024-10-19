try:
    import cupy as cp
except ImportError:
    import numpy as cp
from numtorch.autograd import Tensor
from typing import Union
from numtorch.nn.base import Module

from numtorch.functions.tanh import tanh


class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x: Tensor):
        o = tanh(x)
        return o
