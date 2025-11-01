import numpy as np
import mindspore
from mindspore import ops, nn, Tensor, Parameter

np.random.seed(42)
x = ops.ones(5, mindspore.float32)  # input tensor
y = ops.zeros(3, mindspore.float32)  # expected output
w = Parameter(Tensor(np.random.randn(5, 3), mindspore.float32), name='w') # weight
b = Parameter(Tensor(np.random.randn(3,), mindspore.float32), name='b') # bias

def function(x, y, w, b):
    z = ops.matmul(x, w) + b
    loss = ops.binary_cross_entropy_with_logits(z, y, ops.ones_like(z), ops.ones_like(z))
    return loss


def my_fun(x, y, w, b):
    z = ops.matmul(x, w) + b
    return z


print(my_fun(x, y, w, b))
grad_fn = mindspore.grad(my_fun, (2,3))
grads = grad_fn(x, y, w, b)
print(f"the gradient is: {grads}")
