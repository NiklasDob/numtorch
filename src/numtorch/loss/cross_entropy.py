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
        softmax_probs = self.softmax(predictions)

        # Extract the probabilities for the correct class (as per targets)
        correct_class_probs = softmax_probs[cp.arange(targets._data.shape[0]), targets._data]

        # Compute the negative log-likelihood loss
        loss_value = -cp.log(correct_class_probs + 1e-9)  # Add epsilon for numerical stability
        loss = Tensor(cp.mean(loss_value), children=(predictions,), op="cross_entropy")

        def _backward():
            # Gradient of cross-entropy with respect to softmax probabilities
            grad = softmax_probs
            grad[cp.arange(targets._data.shape[0]), targets._data] -= 1
            grad /= targets._data.shape[0]

            # Propagate gradients back to the predictions
            predictions.grad += grad

        loss._backward = _backward
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
    # Example usage
    # Predictions are of shape (N, C), where N is the number of samples and C is the number of classes
    predictions = Tensor(cp.array([[2.0, 1.0, 0.1], [1.0, 3.0, 0.1], [2.0, 2.0, 2.0]]))
    # Targets are of shape (N,) containing the correct class indices
    targets = Tensor(cp.array([0, 1, 2]), dtype=int)

    # Instantiate the cross-entropy loss layer
    loss_fn = CrossEntropyLoss()

    # Compute the loss
    loss = loss_fn.forward(predictions, targets)
    print("Loss:", loss)

    # Backward pass
    loss.backward()
    print("Gradients of predictions:\n", predictions.grad)
