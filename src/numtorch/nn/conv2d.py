try:
    import cupy as cp
except ImportError:
    import numpy as cp
import numtorch as nt
from typing import Tuple, Union
from numtorch.nn.base import Module


class Conv2D(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: int = 1,
        padding: int = 0,
    ):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights and bias
        self.weight = nt.Tensor(
            cp.random.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]), requires_grad=True
        )
        self.bias = nt.Tensor(cp.random.randn(out_channels), requires_grad=True)

    def forward(self, x: nt.Tensor):
        b, c, h, w = x.shape  # Batch size, input channels, height, width
        kh, kw = self.kernel_size  # Kernel height, kernel width

        # Output height and width calculation with respect to stride and padding
        oh = (h - kh + 2 * self.padding) // self.stride + 1
        ow = (w - kw + 2 * self.padding) // self.stride + 1

        # Apply padding if required
        if self.padding > 0:
            x = nt.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), "constant")

        # Initialize the output tensor
        out = nt.Tensor(cp.zeros((b, self.out_channels, oh, ow)), requires_grad=True)

        # Perform the convolution operation
        for i in range(oh):
            for j in range(ow):
                i_start = i * self.stride
                j_start = j * self.stride
                i_end = i_start + kh
                j_end = j_start + kw

                # Extract the patch from the input and apply the convolution
                x_patch = x[:, :, i_start:i_end, j_start:j_end]  # Shape (batch, in_channels, kh, kw)
                out[:, :, i, j] = out[:, :, i, j] + (
                    nt.sum(
                        self.weight.reshape(1, self.out_channels, c, kh, kw) * x_patch.reshape(b, 1, c, kh, kw),
                        axis=(2, 3, 4),
                    )  # Perform element-wise multiplication
                    + self.bias
                )  # Add bias to the result

        return out


if __name__ == "__main__":
    in_channel = 3
    x = nt.Tensor(cp.random.random((16, in_channel, 5, 5)))
    layer = Conv2D(in_channel, 3)
    y = layer(x)
    # TODO: Figure out why this does not work. Probably something with the reshaping in the broadcast_to i would guess
    y.backward()
