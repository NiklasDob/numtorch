import pytest
import numpy as np
from numtorch.autograd.value import Value


def test_addition():
    a = Value(3.0)
    b = Value(4.0)
    c = a + b
    assert c.value == 7.0


def test_subtraction():
    a = Value(5.0)
    b = Value(2.0)
    c = a - b
    assert c.value == 3.0


def test_multiplication():
    a = Value(3.0)
    b = Value(4.0)
    c = a * b
    assert c.value == 12.0


def test_negation():
    a = Value(5.0)
    c = -a
    assert c.value == -5.0


def test_power():
    a = Value(2.0)
    b = a**3
    assert b.value == 8.0


def test_division():
    a = Value(8.0)
    b = Value(2.0)
    c = a / b
    assert c.value == 4.0


def test_radd():
    a = Value(3.0)
    b = 4.0 + a
    assert b.value == 7.0


def test_rsub():
    a = Value(3.0)
    b = 10.0 - a
    assert b.value == 7.0


def test_rmul():
    a = Value(5.0)
    b = 3.0 * a
    assert b.value == 15.0


def test_rtruediv():
    a = Value(4.0)
    b = 12.0 / a
    assert b.value == 3.0


def test_backward_addition():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    c.grad = 1.0
    c.backward()
    assert a.grad == 1.0
    assert b.grad == 1.0


def test_backward_multiplication():
    a = Value(3.0)
    b = Value(4.0)
    c = a * b
    c.grad = 1.0
    c.backward()
    assert a.grad == 4.0
    assert b.grad == 3.0


def test_backward_power():
    a = Value(2.0)
    b = a**3
    b.grad = 1.0
    b.backward()
    assert pytest.approx(a.grad, 0.00001) == 12.0  # 3 * 2^(3-1) = 12


def test_backward_division():
    a = Value(10.0)
    b = Value(2.0)
    c = a / b
    c.backward()
    assert pytest.approx(a.grad, 0.00001) == 0.5  # derivative is 1/b
    assert pytest.approx(b.grad, 0.00001) == -2.5  # derivative is -a / (b^2)


def test_backward_power_with_value():
    a = Value(2.0)
    b = Value(5.0)
    c = a**b
    c.backward()
    # dc/da = b * a^(b-1) = 5.0 * 2.0^(5.0 - 1) = 80
    # dc/db = a**b * log(a) = 2 ** 5 * log(2) =
    assert pytest.approx(a.grad, 0.00001) == 80.0
    assert pytest.approx(b.grad, 0.00001) == 2.0**5 * np.log(2)


if __name__ == "__main__":
    # test_backward_addition()
    # test_backward_multiplication()
    # test_backward_power()
    # test_backward_division()
    test_backward_power_with_value()
