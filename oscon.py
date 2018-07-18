"""
Livecoding Madness: Let's Build a Deep Learning Library
Joel Grus
@joelgrus
research engineer, Allen Institute for AI
author, _Data Science from Scratch_
#OSCON 2018
"""

# 0. the problem
"""
print the numbers 1 to 100, except
* if the number is divisible by 3, print "fizz"
* if the nubmer is divisible by 5, print "buzz"
* if the nubmer is divisible by 15, print "fizzbuzz"
"""

# 1. Tensors
"""
a tensor is just a n-dimensional array
"""
from numpy import ndarray as Tensor  # type
from numpy import array as tensor    # constructor functino

#2. Loss Function
"""
a loss function shows how good or bad our predictions are
"""
import numpy as np

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class SSE(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)

# 3. Layers
from typing import Iterator, Tuple

class Layer:
    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, gradient: Tensor) -> Tensor:
        raise NotImplementedError

    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        yield from []


class Linear(Layer):
    """
    computes outputs = inputs @ w + b
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        self.w = np.random.randn(input_size, output_size)
        self.b = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        inputs are (batch_size, input_size)
        """
        # save the inputs for the backward pass
        self.inputs = inputs
        return inputs @ self.w + self.b

    def backward(self, gradient: Tensor) -> Tensor:
        """
        if y = f(ab), then dy/da = f'(ab) * b
        """
        self.b_grad = np.sum(gradient, axis=0)
        self.w_grad = self.inputs.T @ gradient
        return gradient @ self.w.T

    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        yield self.w, self.w_grad
        yield self.b, self.b_grad


class Tanh(Layer):
    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return np.tanh(inputs)

    def backward(self, gradient: Tensor) -> Tensor:
        tanh = np.tanh(self.inputs)
        tanh_prime = 1 - tanh ** 2
        return tanh_prime * gradient

# 4. Neural Networks
"""
a neural network is just a collection of layers.
in fact, it's just a layer itself
"""
from typing import List

class NeuralNet(Layer):
    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs


    def backward(self, gradient: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            yield from layer.params_and_grads()

# 5. Optimizers
class Optimizer:
    def update(self, net: NeuralNet) -> None:
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr: float) -> None:
        self.lr = lr

    def update(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad

# 6. Fizz Buzz
"""
print the numbers 1 to 100, except
* if the number is divisible by 3, print "fizz"
* if the nubmer is divisible by 5, print "buzz"
* if the nubmer is divisible by 15, print "fizzbuzz"
"""

def binary_encode(x: int) -> List[int]:
    """
    encode x as a 10-digit binary number
    using some bitwise arithmetic voodoo
    """
    return [x >> i & 1 for i in range(10)]

def fizz_buzz_encode(x: int) -> List[int]:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]


x_train = tensor([binary_encode(i) for i in range(101, 1024)])
y_train = tensor([fizz_buzz_encode(i) for i in range(101, 1024)])

HIDDEN_SIZE = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10_000

net = NeuralNet([
    Linear(input_size=10, output_size=HIDDEN_SIZE),
    Tanh(),
    Linear(input_size=HIDDEN_SIZE, output_size=4)
])

loss = SSE()
optimizer = SGD(lr=LEARNING_RATE)

starts = np.arange(0, len(x_train), BATCH_SIZE)

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    np.random.shuffle(starts)

    for start in starts:
        end = start + BATCH_SIZE

        inputs = x_train[start:end]
        actual = y_train[start:end]

        predicted = net.forward(inputs)

        epoch_loss += loss.loss(predicted, actual)
        gradient = loss.gradient(predicted, actual)

        net.backward(gradient)
        optimizer.update(net)

    print(epoch, epoch_loss)

# do fizz buzz for 1 to 100
correct = 0

for x in range(1, 101):
    inputs = tensor([binary_encode(x)])
    predicted = net.forward(inputs)[0]

    predicted_idx = np.argmax(predicted)
    actual_idx = np.argmax(fizz_buzz_encode(x))

    if predicted_idx == actual_idx:
        correct += 1

    labels = [str(x), "fizz", "buzz", "fizzbuzz"]

    print(x, labels[predicted_idx], labels[actual_idx])

print(correct, "/ 100")
