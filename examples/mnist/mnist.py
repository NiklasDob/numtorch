from numtorch.datasets import load_mnist
from numtorch.nn import MLP
from numtorch.autograd import Tensor


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_mnist()

    model = MLP(784, [128, 64], 10)
    loss = model(x_train)
    loss.backward()
    print(loss)
