import numpy as np


class Value:
    def __init__(self, value, children=(), op=None):
        self.value = value
        self.grad = 0.0
        self._backward = lambda: None
        self._children = children
        self.op = op

    def __repr__(self) -> str:
        return f"Value({self.value})"

    def backward(self):
        self.grad = 1.0

        topo = []
        seen = set()

        def build_topo_sort(child):
            if child in seen:
                return
            seen.add(child)
            for c in child._children:
                build_topo_sort(c)
            topo.append(child)

        build_topo_sort(self)

        for node in reversed(topo):
            node._backward()

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value + other.value, children=(self, other), op="+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._set_backward(_backward)

        return out

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value * other.value, children=(self, other), op="*")

        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad

        out._set_backward(_backward)

        return out

    def __pow__(self, other):

        if isinstance(other, (int, float)):
            out = Value(self.value**other, children=(self,), op=f"**{other}")

            def _backward():
                self.grad += (other * self.value ** (other - 1)) * out.grad

            out._set_backward(_backward)

            return out
        elif isinstance(other, Value):
            out = Value(self.value**other.value, children=(self, other), op=f"**{other}")

            # NOTE: Imagine the current node is x and the other node is y. The output node is z.
            # z = x ** y
            # dz/dx = y * x ** (y - 1)
            # dz/dy = x ** y * log(x)
            def _backward():
                self.grad += (other.value * self.value ** (other.value - 1)) * out.grad
                other.grad += out.value * (np.log(self.value) * out.grad)

            out._set_backward(_backward)

            return out

        else:
            raise NotImplementedError(f"Only int, float and Value are supported for power operation. Not {type(other)}")

    def __neg__(self):
        out = Value(-self.value, children=(self,), op="-")

        def _backward():
            self.grad *= -1.0

        out._set_backward(_backward)

        return out

    def __radd__(self, other):  # other + self
        return self + other

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other ** (-1)

    def __rtruediv__(self, other):  # other / self
        return other * self ** (-1)
