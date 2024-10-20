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
# prints [ 6.  9. 12.]
