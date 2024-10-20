try:
    import cupy as cp
except ImportError:
    print("Could not import cupy, falling back to numpy...")
    import numpy as cp

from numtorch.autograd import Tensor
from typing import Union

from numtorch.nn.base import Module


class CrossEntropyLoss(Module):
    def __init__(self):
        pass

    def forward(self, predictions: Tensor, targets: Tensor):
        """
        Computes the cross-entropy loss between the predictions and targets.

        :param predictions: Tensor of shape (N, C) where N is the number of samples and C is the number of classes.
        :param targets: Tensor of shape (N,) where each value is the correct class index for each sample.
        :return: Scalar Tensor representing the cross-entropy loss.
        """

        # Apply the softmax function to convert predictions to probabilities
        # p_j = e^x_j / \sum_i e^x_i
        softmax_probs = self.softmax(predictions)

        # Extract the probabilities for the correct class (as per targets)
        # p_jc
        correct_class_probs = softmax_probs[cp.arange(targets._data.shape[0]), targets._data]

        # Compute the negative log-likelihood loss
        #  -log(p_jc)
        loss_value = -cp.log(correct_class_probs + 1e-9)  # Add epsilon for numerical stability
        loss = Tensor(
            cp.mean(loss_value),
            requires_grad=predictions._requires_grad,
            dtype=predictions.dtype,
            children=(predictions,),
            op="cross_entropy",
        )

        def _backward():
            # Gradient of cross-entropy with respect to softmax probabilities
            grad = softmax_probs
            grad[cp.arange(targets._data.shape[0]), targets._data] -= 1
            grad /= targets._data.shape[0]

            # Propagate gradients back to the predictions
            predictions.grad += grad

        loss._set_backward(_backward)
        return loss

    def softmax(self, x: Tensor) -> cp.ndarray:
        """
        Applies the softmax function to the input Tensor.
        :param x: Tensor of shape (N, C)
        :return: Softmax probabilities as a numpy/cupy ndarray of the same shape.
        """
        exp_vals = cp.exp(x._data - cp.max(x._data, axis=1, keepdims=True))  # Subtract max for numerical stability
        return exp_vals / cp.sum(exp_vals, axis=1, keepdims=True)


if __name__ == "__main__":
    predictions = Tensor(cp.array([[2.0, 1.0, 0.1], [1.0, 3.0, 0.1], [2.0, 2.0, 2.0]]))
    targets = Tensor(cp.array([0, 1, 2]), dtype=int)

    loss_fn = CrossEntropyLoss()

    loss = loss_fn(predictions, targets)
    print("Loss:", loss)

    loss.backward()
    print("Gradients of predictions:\n", predictions.grad)
