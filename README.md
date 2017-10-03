A minimalistic example of reverse-mode automatic differentiation
imitating the PyTorch interface. We create a bunch of `Variable`
objects, do simple operations with them then and call `.backward()` to
compute the gradient components, accessible via `.grad`:

```python
x1 = Variable(1.0)
x2 = Variable(2.0)

a1 = x1*x2
a2 = x1/x2

b1 = a1/a2
b2 = a1*a2

y = b1-b2

y.backward()

# -2.0 4.0`
print x1.grad, x2.grad
```
