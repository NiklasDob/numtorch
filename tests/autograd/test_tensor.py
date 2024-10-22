import pytest

try:
    import cupy as cp
except ImportError:
    import numpy as cp
from numtorch.autograd import Tensor
import numtorch as nt


def test_tensor_creation():
    # Test tensor creation with a list
    t1 = Tensor([1, 2, 3], requires_grad=True)
    assert isinstance(t1, Tensor)
    assert t1.shape == (3,)
    assert t1._requires_grad is True
    assert cp.array_equal(t1._data, cp.array([1, 2, 3]))

    # Test tensor creation with cupy array
    t2 = Tensor(cp.array([4, 5, 6]), requires_grad=False)
    assert t2.shape == (3,)
    assert t2._requires_grad is False
    assert cp.array_equal(t2._data, cp.array([4, 5, 6]))


def test_tensor_addition():
    t1 = Tensor([1, 2, 3], requires_grad=True)
    t2 = Tensor([4, 5, 6], requires_grad=True)

    t3 = t1 + t2

    assert cp.array_equal(t3._data, cp.array([5, 7, 9]))
    assert t3._requires_grad is True

    # Test backward propagation
    t3.backward()
    assert cp.array_equal(t1.grad, cp.array([1, 1, 1]))
    assert cp.array_equal(t2.grad, cp.array([1, 1, 1]))


def test_tensor_multiplication():
    t1 = Tensor([1, 2, 3], requires_grad=True)
    t2 = Tensor([4, 5, 6], requires_grad=True)

    t3 = t1 * t2

    assert cp.array_equal(t3._data, cp.array([4, 10, 18]))
    assert t3._requires_grad is True

    # Test backward propagation
    t3.backward()
    assert cp.array_equal(t1.grad, cp.array([4, 5, 6]))
    assert cp.array_equal(t2.grad, cp.array([1, 2, 3]))


def test_tensor_negation():
    t1 = Tensor([1, 2, 3], requires_grad=True)
    t2 = -t1

    assert cp.array_equal(t2._data, cp.array([-1, -2, -3]))

    # Test backward propagation
    t2.backward()
    assert cp.array_equal(t1.grad, cp.array([-1, -1, -1]))


def test_tensor_power():
    t1 = Tensor([1, 2, 3], requires_grad=True)
    t2 = t1**2

    assert cp.array_equal(t2._data, cp.array([1, 4, 9]))

    # Test backward propagation
    t2.backward()
    assert cp.array_equal(t1.grad, cp.array([2, 4, 6]))


def test_tensor_reshape():
    t1 = Tensor([[1, 2], [3, 4]], requires_grad=True)
    t2 = t1.reshape(4)

    assert t2.shape == (4,)
    assert cp.array_equal(t2._data, cp.array([1, 2, 3, 4]))

    # Test backward propagation
    t2.backward()
    assert cp.array_equal(t1.grad, cp.array([[1, 1], [1, 1]]))


def test_tensor_broadcasting():
    t1 = Tensor([[1, 2], [3, 4]], requires_grad=True)
    t2 = Tensor([1, 1], requires_grad=True)

    t3 = t1 + t2

    assert cp.array_equal(t3._data, cp.array([[2, 3], [4, 5]]))
    assert t3._requires_grad is True

    # Test backward propagation
    t3.backward()
    assert cp.array_equal(t1.grad, cp.array([[1, 1], [1, 1]]))
    assert cp.array_equal(t2.grad, cp.array([2, 2]))


def test_tensor_indexing():
    t1 = Tensor([1, 2, 3], requires_grad=True)

    t2 = t1[1]

    assert cp.array_equal(t2._data, cp.array(2))

    # Test backward propagation
    t2.backward()
    assert cp.array_equal(t1.grad, cp.array([0, 1, 0]))


def test_tensor_setitem():

    t0 = Tensor([0, 0, 0], requires_grad=True)
    t1 = Tensor([1, 2, 3], requires_grad=True)
    t2 = Tensor([4, 5, 6], requires_grad=True)

    t0[0] = nt.sum(t1 * t2)

    assert cp.array_equal(t0._data, cp.array([4 + 10 + 18, 0, 0]))
    # Test backward propagation
    t3 = t0 * 2
    t3.backward()

    assert cp.array_equal(t2.grad, cp.array([1, 2, 3]) * 2)
    assert cp.array_equal(t1.grad, cp.array([4, 5, 6]) * 2)


def test_tensor_division():
    t1 = Tensor([1, 2, 3], requires_grad=True)
    t2 = Tensor([4, 5, 6], requires_grad=True)

    t3 = t1 / t2

    assert cp.allclose(t3._data, cp.array([1 / 4, 2 / 5, 3 / 6]))

    # Test backward propagation
    t3.backward()
    assert cp.allclose(t1.grad, cp.array([1 / 4, 1 / 5, 1 / 6]))
    assert cp.allclose(t2.grad, cp.array([-1 / 16, -2 / 25, -3 / 36]))


def test_tensor_topo_sort():
    # Check the topological sorting of the graph during backward propagation
    t1 = Tensor([1, 2, 3], requires_grad=True)
    t2 = Tensor([4, 5, 6], requires_grad=True)

    t3 = t1 + t2
    t4 = t3 * t1

    t4.backward()

    assert cp.array_equal(t1.grad, cp.array([6, 9, 12]))
    assert cp.array_equal(t2.grad, cp.array([1, 2, 3]))


if __name__ == "__main__":
    # pytest.main()
    # test_tensor_topo_sort()
    test_tensor_setitem()
