try:
    import cupy as cp
except ImportError:
    import numpy as cp
import numtorch as nt
from typing import Tuple, Union
from numtorch.nn.base import Module

class MaxPooling2D(Module):
    def __init__(
        self,
        kernel_size: Tuple[int, int] = (2, 2),
        stride: Union[int, None] = None,
        padding: int = 0,
    ):
        super(MaxPooling2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size[0]
        self.padding = padding

    def forward(self, x: nt.Tensor):
        b, c, h, w = x.shape
        kh, kw = self.kernel_size

        oh = (h - kh + 2 * self.padding) // self.stride + 1
        ow = (w - kw + 2 * self.padding) // self.stride + 1

        if self.padding > 0:
            x = nt.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), "constant")

        out = nt.Tensor(cp.zeros((b, c, oh, ow)), requires_grad=True)

        for i in range(oh):
            for j in range(ow):
                i_start = i * self.stride
                j_start = j * self.stride
                i_end = i_start + kh
                j_end = j_start + kw

                x_patch = x[:, :, i_start:i_end, j_start:j_end]

                out[:, :, i, j] = nt.max(x_patch, axis=(2, 3))

        return out


if __name__ == "__main__":
    in_channel = 3
    x = nt.Tensor(cp.random.random((16, in_channel, 5, 5)))
    pool_layer = MaxPooling2D(kernel_size=(2, 2), stride=2)
    pooled_output = pool_layer(x)
    # You can print or debug the output if needed
    print(pooled_output.shape)
