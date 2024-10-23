try:
    import cupy as cp
except ImportError:
    import numpy as cp

from tqdm import tqdm
from numtorch.datasets import load_mnist
from numtorch.loss import CrossEntropyLoss
from numtorch.nn import MLP, Conv2D
from numtorch.autograd import Tensor
from numtorch.nn.base import Module
from numtorch.nn.relu import ReLU
from numtorch.nn.sequential import Sequential
from numtorch.nn.sigmoid import Sigmoid
from numtorch.nn.tanh import Tanh
from numtorch.optim.adamw import AdamW
from numtorch.optim.sgd import SGD


def calculate_accuracy(predictions: Tensor, targets: Tensor) -> float:
    predicted_classes = cp.argmax(predictions._data, axis=1)
    correct_predictions = cp.sum(predicted_classes == targets._data)
    accuracy = correct_predictions / targets._data.shape[0]
    return float(accuracy)


class Net(Module):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(in_channels=1, out_channels=3, kernel_size=(3, 3))
        self.mlp = MLP(3 * 26 * 26, [256, 128, 64], 10, activation=ReLU())

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv1(x)
        x = x.reshape(b, -1)
        x = self.mlp(x)
        return x

if __name__ == "__main__":
    cp.random.seed(0)
    (x_train, y_train), (x_test, y_test) = load_mnist()

    learning_rate = 5e-3

    net = Net()
    loss_metric = CrossEntropyLoss()
    optimizer = AdamW(net.parameters(), lr=learning_rate)
    batch_size = 1024
    num_batches = x_train.shape[0] // batch_size
    num_epochs = 100
    for epoch in tqdm(range(num_epochs), desc="Epochs", total=num_epochs):
        for i in tqdm(range(num_batches), desc="Batches", total=num_batches):
            start = i * batch_size
            end = start + batch_size
            x_batch = x_train[start:end].reshape(-1, 1, 28, 28)  # Reshape input for Conv2D layer
            y_batch = y_train[start:end]

            y_pred = net(x_batch)
            loss = loss_metric(y_pred, y_batch)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()


        # Evaluate on test data
        y_pred_test = net(x_test.reshape(-1, 1, 28, 28))  # Reshape test data for Conv2D layer
        loss_test = loss_metric(y_pred_test, y_test)
        accuracy_test = calculate_accuracy(y_pred_test, y_test)
        print(f"Test Loss: {loss_test}, Test Accuracy: {accuracy_test}")