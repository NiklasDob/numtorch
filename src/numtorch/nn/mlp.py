try:
    import cupy as cp
except ImportError:
    import numpy as cp
import copy
from numtorch.autograd import Tensor
from typing import Union, List
from numtorch.nn.sequential import Sequential
from numtorch.nn.linear import Linear
from numtorch.nn.base import Module


class MLP(Module):

    def __init__(self, in_features: int, hidden_features: List[int], out_features: int):
        self.in_features = in_features
        self.hidden_features = copy.deepcopy(hidden_features)
        self.hidden_features.append(out_features)
        self.out_features = out_features
        assert len(hidden_features) > 0

        self.seq = Sequential(
            Linear(in_features, hidden_features[0]),
            *[Linear(hidden_features[i], hidden_features[i + 1]) for i in range(len(hidden_features) - 1)],
            Linear(hidden_features[-1], out_features),
        )

    def forward(self, x: Tensor):
        x = self.seq(x)
        return x
