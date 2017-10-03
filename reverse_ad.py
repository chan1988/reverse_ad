class Variable(object):
    def __init__(self, data):
        self.data = float(data)
        self.grad = 0.0

        self.parents = None

    def __mul__(self, other):
        z = Variable(self.data*other.data)
        z.parents = [(self, other.data), (other, self.data)]
        return z

    def __div__(self, other):
        z = Variable(self.data/other.data)
        z.parents = [(self, 1/other.data), (other, -self.data/(other.data)**2)]
        return z

    def __add__(self, other):
        z = Variable(self.data + other.data)
        z.parents = [(self, 1.0), (other, 1.0)]
        return z

    def __sub__(self, other):
        z = Variable(self.data - other.data)
        z.parents = [(self, 1.0), (other, -1.0)]
        return z

    def backward(self, signal=1.0):
        self.grad = signal
        backprop(self, signal)


def backprop(node, signal):
    if node.parents is None: return
    for p, s in node.parents:
        new_signal = s*signal
        p.grad += new_signal
        backprop(p, new_signal)


x1 = Variable(1.0)
x2 = Variable(2.0)

a1 = x1*x2
a2 = x1/x2

b1 = a1/a2
b2 = a1*a2

y = b1-b2

y.backward()

# -2.0 4.0
print x1.grad, x2.grad
