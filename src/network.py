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
    
    def update_mini_batch(self, mini_batch, eta):
        """
        x
        """
        nabla_b = [np.zeros(shape=b.shape) for b in self.bias]
        nabla_w = [np.zeros(shape=w.shape) for w in self.weight]
        for x, y in mini_batch:
            delta_b, delta_w = self.backprop(x, y)
            nabla_b = [nb_b + del_b for nb_b, del_b in zip(nabla_b, delta_b)]
            nabla_w = [nb_w + del_w for nb_w, del_w in zip(nabla_w, delta_w)]
        self.weight = [w-(eta/len(mini_batch)) * nb_w for w, nb_w in zip(self.weight, nabla_w)]
        self.bias = [b-(eta/len(mini_batch)) * nb_b for b, nb_b in zip(self.bias, nabla_b)]