import numpy as np
from typing import TYPE_CHECKING, Union
if TYPE_CHECKING:
    from numpy.typing import NDArray
from neurons import SigmoidNeuron

def sigmoid(z):
    """
    x
    """
    return 1/(1 + np.exp(-z))

def sigmoid_prime(z):
    """
    x
    """
    sigmoid = sigmoid(z)
    return sigmoid * (1 - sigmoid)

class Network:
    """
    x
    """

    def __init__(self, sizes: list):
        """
        x
        """
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.weight: list['NDArray[np.float64]'] = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.bias: list['NDArray[np.float64]'] = [np.random.randn(y, 1) for y in self.sizes[1:]]

    def feedforward(self, a: 'NDArray[np.float64]'):
        """
        x
        """
        for b, w in zip(self.bias, self.weight):
            z = np.dot(w, a) + b
            a = sigmoid(z)
        return a
    
    def backprop(self, x: 'NDArray[np.float64]', y: 'NDArray[np.float64]'):
        """
        x
        """
        zs: list = []
        activations: list = [x]
        a = x
        for b, w in zip(self.bias, self.weight):
            z = np.dot(w, a) + b
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)
        severity = activations[-1] - y
        error_L = severity * sigmoid_prime(zs[-1])

        nabla_b = [np.zeros(shape=b.shape) for b in self.bias]
        nabla_w = [np.zeros(shape=w.shape) for w in self.weight]

        nabla_b[-1] = error_L
        nabla_w[-1] = np.dot(error_L, (activations[-2]).T)

        for l in range(2, self.num_layers):
            error_L = np.dot(self.weight[-l+1].T, error_L) * sigmoid_prime(zs[-l])
            nabla_b[-l] = error_L
            nabla_w[-l] = np.dot(error_L, activations[-l-1].T)
        return (nabla_b, nabla_w)