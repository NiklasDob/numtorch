try:
    import cupy as cp
except ImportError:
    import numpy as cp

from numtorch.autograd import Tensor
from numtorch.nn.base import Module
from typing import Union


class Softmax(Module):
    def __init__(self, axis: int = 1):
        """
        :param axis: The axis along which the softmax function is applied. Default is 1 (rows).
        """
        super(Softmax, self).__init__()
        self.axis = axis

    def forward(self, x: Tensor):
        """
        Applies the softmax function to the input tensor.
        :param x: Tensor of shape (N, C) where N is the number of samples and C is the number of classes.
        :return: Tensor with softmax probabilities of the same shape.
        """
        exp_vals = cp.exp(x._data - cp.max(x._data, axis=self.axis, keepdims=True))
        softmax_probs = exp_vals / cp.sum(exp_vals, axis=self.axis, keepdims=True)
        out = Tensor(softmax_probs, children=(x,), op="softmax")

        def _backward():
            """
            Gradient of softmax: dL/dx = softmax(x) * (1 - softmax(x)) for each element.
            """
            grad = out.grad
            s = softmax_probs

            # Compute the Jacobian matrix for each sample and the gradient using the chain rule
            for i in range(s.shape[0]):
                s_i = s[i].reshape(-1, 1)  # Column vector
                jacobian = cp.diagflat(s_i) - cp.dot(s_i, s_i.T)
                grad_i = grad[i].reshape(1, -1)  # Row vector
                x.grad[i] += cp.dot(grad_i, jacobian).flatten()

        out._backward = _backward
        return out


if __name__ == "__main__":
    # Testing the Softmax class
    x = Tensor(cp.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))
    softmax_layer = Softmax()

    # Apply the softmax function
    y = softmax_layer.forward(x)
    print("Softmax output:\n", y)

    # Backward pass
    y.backward()
    print("Gradient of x:\n", x.grad)
