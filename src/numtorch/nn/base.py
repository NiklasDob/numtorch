from typing import List

from numtorch.autograd.tensor import Tensor


class Module(object):
    def __init__(self):
        super(Module, self).__init__()

    def forward(self, *args):
        raise NotImplementedError("Needs to implement a forward function")

    def parameters(self) -> List[Tensor]:
        params = []
        for k, var in vars(self).items():
            if isinstance(var, Tensor):
                params.append(var)
            elif isinstance(var, Module):
                params.extend(var.parameters())
        return params

    def __call__(self, *args):
        return self.forward(*args)
