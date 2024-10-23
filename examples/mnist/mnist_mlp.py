try:
    import cupy as cp
except ImportError:
    import numpy as cp

from tqdm import tqdm
from numtorch.datasets import load_mnist
from numtorch.loss import CrossEntropyLoss
from numtorch.nn import MLP
from numtorch.autograd import Tensor
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


if __name__ == "__main__":
    cp.random.seed(0)
    (x_train, y_train), (x_test, y_test) = load_mnist()

    learning_rate = 5e-3
    mlp = MLP(784, [256, 128, 64], 10, activation=Tanh())
    net = Sequential(mlp)
    loss_metric = CrossEntropyLoss()

    optimizer = AdamW(net.parameters(), lr=learning_rate)
    batch_size = 512
    num_batches = x_train.shape[0] // batch_size
    for epoch in tqdm(range(100), desc="Epochs"):
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            x_batch = x_train[start:end]
            y_batch = y_train[start:end]

            y_pred = net(x_batch.reshape(-1, 784))
            loss = loss_metric(y_pred, y_batch)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        y_pred_test = net(x_test.reshape(-1, 784))
        loss_test = loss_metric(y_pred_test, y_test)
        accuracy_test = calculate_accuracy(y_pred_test, y_test)
        print(f"Test Loss: {loss_test}, Test Accuracy: {accuracy_test}")
    # print(loss, x_train.grad)
