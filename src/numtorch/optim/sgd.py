try:
    import cupy as cp
except ImportError:
    import numpy as cp

from typing import List
from numtorch.autograd import Tensor
from numtorch.optim.base import Optimizer


class SGD(Optimizer):
    def __init__(self, parameters: List[Tensor], lr: float = 1e-2, maximize: bool = False):
        super().__init__(parameters)
        self.lr = lr
        self.maximize = maximize

    def step(self):
        maximize = 1.0 if self.maximize else -1.0

        for p in self.parameters:
            p._data = p._data + maximize * self.lr * p.grad
