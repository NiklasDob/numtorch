from typing import Tuple, Union
from numtorch.autograd.tensor import Tensor


try:
    import cupy as cp
except ImportError:
    import numpy as cp


def sum(x: Tensor, axis: Union[int, Tuple[int, ...]] = -1, keepdims: bool = False):
    out = Tensor(
        cp.sum(x._data, axis=axis, keepdims=keepdims),
        requires_grad=x._requires_grad,
        dtype=x.dtype,
        children=(x,),
        op="nt.sum",
    )

    # Such that the out.grad is correctly broadcasted in the backward step
    out_grad_shape = cp.array(x.shape)
    if isinstance(axis, int):
        out_grad_shape[axis] = 1
    else:
        for a in axis:
            out_grad_shape[a] = 1

    out_grad_shape = tuple(out_grad_shape.tolist())

    # (10,1)
    def backward():
        # z = a + b
        # dz/da = 1.0
        x.grad += out.grad.reshape(out_grad_shape)

    out._set_backward(backward)

    return out


if __name__ == "__main__":
    x = Tensor(cp.ones((2, 2, 2)), requires_grad=True)
    y = sum(x, axis=(1, 2))
    y = y * y
    print(y.shape)
    y.backward()
    print(x.grad)
