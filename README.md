# NumTorch

**NumTorch** is a personal project aimed at recreating fundamental features of PyTorch using only NumPy/CuPy. The goal of the project is to gain a deeper understanding of the inner workings of PyTorch. This autograd implementation is inspired [Micrograd](https://github.com/karpathy/micrograd).

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction

NumTorch is a lightweight framework inspired by PyTorch. It aims to implement core functionalities such as:
- **Tensor operations** using NumPy/CuPy as the backend.
- **Automatic differentiation** to facilitate gradient-based optimization, providing a similar experience to PyTorch’s autograd.
- **Modules** Linear layer just like in PyTorch.

This project serves as an educational tool for exploring the mechanics behind popular deep learning libraries by building them from the ground up using only NumPy.

## Features

- **Tensors**: Implement basic operations such as addition, multiplication, and reshaping.
- **Automatic Differentiation**: Support for computing gradients with respect to tensor operations.
- **Chain Rule Implementation**: Enables backpropagation for training simple neural networks.
- **Modular Design**: Each component is designed to be as simple and modular as possible for learning purposes.

## Installation

NumTorch does not require installation. Simply clone the repository and use it directly in your Python environment:

```bash
git clone https://github.com/NiklasDob/numtorch
cd numtorch
```

Ensure that you have Python and NumPy installed in your environment:

```bash
pip install numpy
```

## Usage

To use NumTorch, import the library into your Python script:

```python
import numtorch as nt

# Create tensors
a = nt.Tensor([1, 2, 3])
b = nt.Tensor([4, 5, 6])

# Perform operations
c = a + b
d = c * a

# Backpropagation
d.backward()
print(a.grad)  # Gradient of 'a' with respect to 'd'
```

# Simple training on mnist
See in `examples/mnist/mnist.py` for more details.

```python

(x_train, y_train), (x_test, y_test) = load_mnist()

learning_rate = 1e-1
mlp = MLP(784, [256, 128, 64, 32], 10, activation=Tanh())
net = Sequential(mlp)
loss_metric = CrossEntropyLoss()

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

        for param in net.parameters():
            param._data = param._data - learning_rate * param.grad

        # Reset gradients
        #  TODO: Write an optimizer class for this
        for param in net.parameters():
            param.grad = cp.zeros_like(param._data)

    y_pred_test = net(x_test.reshape(-1, 784))
    loss_test = loss_metric(y_pred_test, y_test)
    accuracy_test = calculate_accuracy(y_pred_test, y_test)
    print(f"Test Loss: {loss_test}, Test Accuracy: {accuracy_test}")
```


The above code snippet demonstrates creating tensors, performing operations, and computing gradients using backpropagation.

## Examples

NumTorch comes with a set of example scripts that demonstrate the library’s features:

- **Simple Gradient Calculation**: A basic example of computing gradients using automatic differentiation.
- **Linear Regression**: Training a simple linear model with gradient descent.
- **Neural Network**: Implementing and training a basic neural network using NumTorch’s autograd functionality.

Refer to the `examples/` folder for more details.

## Contributing

Contributions are welcome! If you are interested in adding features, improving code quality, or fixing bugs, feel free to submit a pull request.

Please follow the guidelines below:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Add new feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to explore, learn, and contribute to **NumTorch**! If you have any questions or feedback, please open an issue on the GitHub repository.