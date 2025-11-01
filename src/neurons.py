import numpy as np

class Perceptron:
    """
    x
    """

    def __init__(self, N):
        """
        x
        """
        self.weight = np.random.randn(N, 1)
        self.bias = np.random.randn(1, 1)

    def activation_function(self, z) -> float:
        """
        x
        """
        return 1.0 if z > 0 else 0.0
    
    def forward(self, x):
        """
        x
        """
        weighted_sum = np.dot(self.weight.T, x) + self.bias
        return Perceptron.activation_function(z=weighted_sum)