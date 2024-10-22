from typing import Tuple, Union
from numtorch.autograd.tensor import Tensor


try:
    import cupy as cp
except ImportError:
    import numpy as cp


def pad(x: Tensor, pad_width: Union[int, Tuple[Tuple[int, int], ...]], mode="constant", **kwargs):
    # Apply the padding using cp.pad (can be NumPy or CuPy depending on the backend)
    padded_array = cp.pad(x._data, pad_width, mode=mode, **kwargs)

    # Create a new tensor that stores the result of padding
    out = Tensor(
        padded_array,
        requires_grad=x._requires_grad,
        dtype=x.dtype,
        children=(x,),
        op="nt.pad",
    )

    # Function to undo the padding during the backward pass
    def _backward():
        grad = out.grad

        slices = []
        for i, (pad_before, pad_after) in enumerate(pad_width):
            slices.append(slice(pad_before, grad.shape[i] - pad_after))

        # Apply the unpadded slices to the gradient
        unpadded_grad = grad[tuple(slices)]

        x.grad += unpadded_grad

    # Set the backward function for this operation
    out._set_backward(_backward)

    return out


if __name__ == "__main__":
    x = Tensor(cp.ones((2, 2, 2)), requires_grad=True)
    y = pad(x, ((0, 0), (0, 0), (2, 2)), mode="constant", constant_values=0)
    print(y.shape)
    y.backward()
    print(x.grad)
