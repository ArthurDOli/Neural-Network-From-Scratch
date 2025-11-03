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

class Network:
    """
    x
    """

    def __init__(self, sizes: list):
        """
        x
        """
        self.sizes = sizes
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
    
    def backprop(self, x: 'NDArray[np.float64]'):
        """
        x
        """
        pass