try:
    import cupy as cp
except ImportError:
    import numpy as cp

from numtorch.autograd import Tensor
from typing import Union
from numtorch.functions import sigmoid
from numtorch.nn.base import Module


class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x: Tensor):
        sigmoid_out = sigmoid(x)
        return sigmoid_out
