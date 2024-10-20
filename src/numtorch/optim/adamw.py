try:
    import cupy as cp
except ImportError:
    import numpy as cp

from typing import List, Tuple
from numtorch.autograd import Tensor
from numtorch.optim.base import Optimizer


class AdamW(Optimizer):

    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 1e-2,
        eps: float = 1e-08,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.999),
        maximize: bool = False,
        amsgrad: bool = False,
    ):
        super().__init__(parameters)
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay
        self.betas = betas
        self.maximize = maximize

        self.m = 0.0
        self.v = 0.0
        self.v_max = 0.0
        self.amsgrad = amsgrad

    def step(self):
        maximize = -1.0 if self.maximize else 1.0
        for p in self.parameters:
            g = maximize * p.grad
            theta = p._data - self.lr * self.weight_decay * p._data
            m = self.betas[0] * self.m + (1 - self.betas[0]) * g
            v = self.betas[1] * self.v + (1 - self.betas[1]) * g**2
            m_hat = m / (1 - self.betas[0])
            v_hat = v / (1 - self.betas[1])

            if self.amsgrad:
                self.v_max = cp.maximum(self.v_max, v_hat)
                p._data = theta - self.lr * m_hat / (cp.sqrt(self.v_max) + self.eps)
            else:
                p._data = theta - self.lr * m_hat / (cp.sqrt(v_hat) + self.eps)
