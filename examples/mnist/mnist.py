from numtorch.datasets import load_mnist
from numtorch.loss import CrossEntropyLoss
from numtorch.nn import MLP
from numtorch.autograd import Tensor
from numtorch.nn.relu import ReLU
from numtorch.nn.sequential import Sequential
from numtorch.nn.sigmoid import Sigmoid


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_mnist()

    mlp = MLP(784, [128, 64], 10, activation=ReLU())
    net = Sequential(mlp)
    loss_metric = CrossEntropyLoss()
    y_pred = net(x_train.reshape(-1, 784))
    loss = loss_metric(y_pred, y_train)
    loss.backward()
    print(loss, x_train.grad)
