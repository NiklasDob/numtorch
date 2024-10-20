try:
    import cupy as cp
except ImportError:
    import numpy as cp

from typing import List
from numtorch.autograd.tensor import Tensor


class Optimizer:

    parameters: List[Tensor]  # List of parameters

    def __init__(self, parameters: List[Tensor]):
        self.parameters = parameters
        for p in self.parameters:
            p._requires_grad = True

    def step(self):
        raise NotImplementedError("The step functions needs to be implemented")

    def zero_grad(self):
        for p in self.parameters:
            p.grad = cp.zeros_like(p.grad)
