import numpy as np
from typing import Union, TYPE_CHECKING
if TYPE_CHECKING:
    from numpy.typing import NDArray

class Perceptron:
    """
    Implements the fundamental binary decision mechanism of the Perceptron articifial neuron.

    The Perceptron determines its output (0 or 1) based on whether the weighted sum of inputs 
    exceeds a threshold (represented by the bias).
    """

    def __init__(self, num_inputs: int):
        """
        Initialized the Perceptron's parameters (weights and bias) randomly.

        Args:
            num_imputs: The number of binary inputs the perceptron will receives.
        """
        self.weight: 'NDArray[np.float64]' = np.random.randn(num_inputs, 1)
        self.bias: 'NDArray[np.float64]' = np.random.randn(1, 1)

    def activation_function(self, z: Union[float | 'NDArray[np.float64]']) -> float:
        """
        The Perceptron's non-smooth binary step activation function.

        Outputs 1.0 if the weighted input (z) is strictly greater than zero, otherwise outputs
        0.0.

        Args:
            z: The weighted input (z = w * x + b)
        
        Returns:
            The binary output (0.0 or 1.0)
        """
        z_scalar = z.item() if isinstance(z, np.ndarray) else z
        return 1.0 if z_scalar > 0 else 0.0
    
    def forward(self, x: 'NDArray[np.float64]') -> float:
        """
        Performs the forward propagation step: calculates t he weighted input and applies the
        activation function to produce a binary output.

        Args:
            x: The input vector (N, 1). For the Perceptron, inputs are binary.

        Returns:
            The final binary output (0.0 or 1.0) of the neuron.
        """
        weighted_sum = np.dot(self.weight.T, x) + self.bias
        return self.activation_function(z=weighted_sum)

class SigmoidNeuron:
    """
    Implementes the core logic of an artificial neuron using Sigmoid activation function.

    This neuron resolves the limitations of the binary Perceptron by providin a continuous,
    dufferentiable output.
    """

    def __init__(self, num_inputs: int):
        """
        Initialized the neuron parameters (weights and bias) randomly.

        Args:
            num_imputs: The number of inputs the neuron will receives.
        """
        self.weights: 'NDArray[np.float64]' = np.random.randn(num_inputs, 1)
        self.bias: 'NDArray[np.float64]' = np.random.randn(1, 1)

    def activation_function(self, z: Union[float | 'NDArray[np.float64]']) -> float:
        """
        Calculates the Sigmoid activation function.

        Args:
            z: The weighted input (w * x + b).

        Returns:
            The output activation (a), a continuous value between 0 and 1.
        """
        sigmoid: 'NDArray[np.float64]' = 1/(1 + np.exp(-z))
        return sigmoid.item() if isinstance(sigmoid, np.ndarray) else sigmoid
    
    def forward(self, x: 'NDArray[np.float64]') -> float:
        """
        Performs the forward propagation step for the neuron.

        Calculates the weighted input and aplies the sigmoid activation.

        Args:
            x: The input vector (N, 1).

        Returns:
            The final activation output (a) of the neuron.
        """
        weighted_input = np.dot(self.weights.T, x) + self.bias
        return self.activation_function(z=weighted_input)