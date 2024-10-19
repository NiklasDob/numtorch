try:
    import cupy as cp
except ImportError:
    import numpy as cp
from numtorch.autograd import Tensor
from typing import Union
from numtorch.nn.base import Module

from numtorch.functions.relu import relu


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x: Tensor):
        relu_out = relu(x)
        return relu_out
