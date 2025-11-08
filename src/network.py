import numpy as np
from typing import TYPE_CHECKING, Union, List, Tuple
if TYPE_CHECKING:
    from numpy.typing import NDArray
from neurons import SigmoidNeuron
import random

def sigmoid(z: 'NDArray[np.float64]') -> 'NDArray[np.float64]':
    """
    Computes the sigmoid activation function

    Clips input values to [-709, 709] to prevent overflow in exp(-z) calculation, since exp(709) is near the maximum float64 value

    Args:
        z: Input array (weighted sum)
    """
    return 1/(1 + np.exp(-np.clip(z, -709, 709)))

def sigmoid_prime(z: 'NDArray[np.float64]') -> 'NDArray[np.float64]':
    """
    Computes the derivative of the sigmoid activation function

    Uses the property: sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))

    Args:
        z: Input array (weighted sum)
    """
    sig = sigmoid(z)
    return sig * (1 - sig)

class Network:
    """
    A fully-connected feedforward neural network with Sigmoid activation and L2 regularization

    Supports dropout regularization during training and implements stochastic gradient descent with backpropagation for training
    """

    def __init__(self, sizes: List[int]):
        """
        Initializes the network architecture with random weights and biases

        Uses 'He' initialization for weights: np.sqrt(2.0/(input_size + output_size)) to improve convergence in deep networks
        with ReLU-like activations

        Args:
            sizes: List of integers specifying number of neurons in each layers.
                Example: [784, 128, 10] creates a network with 784 inputs, one hidden layer with 128 neurons and 10 outputs
        """
        self.sizes = sizes
        self.lambda_reg = 0.001
        self.num_layers = len(sizes)
        self.weight: list['NDArray[np.float64]'] = [np.random.randn(y, x) * np.sqrt(2.0/(x+y)) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.bias: list['NDArray[np.float64]'] = [np.random.randn(y, 1) for y in self.sizes[1:]]

    def feedforward(self, a: 'NDArray[np.float64]', training: bool = True) -> 'NDArray[np.float64]':
        """
        Performs forward propagation through the network

        Applies dropout to hidden layers during training for regularization. No dropout is applied during evaluation

        Args:
            a: Input activation vector from the previous layers
            training: If 'True', aplies dropout to hidden hayers. Set to False during evaluation
        """
        for b, w in zip(self.bias[:-1], self.weight[:-1]):
            z = np.dot(w, a) + b
            a = sigmoid(z)
            if training:
                dropout_mask = np.random.binomial(1, 0.8, size=a.shape) / 0.8
                a *= dropout_mask
        z_last = np.dot(self.weight[-1], a) + self.bias[-1]
        a_last = sigmoid(z_last)
        return a_last
    
    def backprop(self, x: 'NDArray[np.float64]', y: 'NDArray[np.float64]') -> Tuple[List['NDArray[np.float64]'], List['NDArray[np.float64]']]:
        """
        Computes gradients using backpropagation algorithm

        Args:
            x: Input sample/feature vector (shape: input_size x 1)
            y: One-hot encoded true label vector (shape: output_size x 1)
        """
        zs: List['NDArray[np.float64]'] = []
        activations: List['NDArray[np.float64]'] = [x]
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
    
    def update_mini_batch(self, mini_batch: List[Tuple['NDArray[np.float64]', 'NDArray[np.float64]']], eta: float) -> None:
        """
        Updates network weights and biases using gradients from a mini-batch

        Implements L2 regularization (weight decay) and stochastic gradient descent

        Learning rate is applied per-sample (divided by batch size)

        Args:
            mini_batch: List of tuples (x, y) where 'x' is input vector and 'y' is one-hot encoded label
            eta: Learning rate (step size) for gradient descent
        """
        nabla_b = [np.zeros(shape=b.shape) for b in self.bias]
        nabla_w = [np.zeros(shape=w.shape) for w in self.weight]
        for x, y in mini_batch:
            delta_b, delta_w = self.backprop(x, y)
            nabla_b = [nb_b + del_b for nb_b, del_b in zip(nabla_b, delta_b)]
            nabla_w = [nb_w + del_w for nb_w, del_w in zip(nabla_w, delta_w)]
        m_batch = len(mini_batch)
        lambda_factor = eta * self.lambda_reg / m_batch
        self.weight = [(1 - lambda_factor) * w - (eta/m_batch) * nb_w for w, nb_w in zip(self.weight, nabla_w)]
        self.bias = [b-(eta/len(mini_batch)) * nb_b for b, nb_b in zip(self.bias, nabla_b)]

    def evaluate(self, test_data: List[Tuple['NDArray[np.float64]', int]]) -> int:
        """
        Evaluates network accuracy on test data

        Args:
            test_data: List of tuples (x, y) where 'x' is input vector and 'y' is integer label index
        """
        results = 0
        for x, y in test_data:
            prevision = self.feedforward(x, training=False)
            prevision_index = np.argmax(prevision)
            if prevision_index == y:
                results += 1
        return results

    def SGD(self, 
            training_data: List[Tuple['NDArray[np.float64]', 'NDArray[np.float64]']], 
            epochs: int, 
            mini_batch_size: int, 
            eta: float, 
            test_data: List[Tuple['NDArray[np.float64]', int]] = None) -> None:
        """
        Trains the network using Stochastic Gradient Descent

        Implements learning rate decay (halves every 20 epochs) and prints progress after each epoch

        Args:
            training_data: List of tuples (x, y) where 'x' is input vector and 'y' is integer label index
            epochs: Number of training epochs (full passes through training data)
            mini_batch_size: Number of samples per mini_batch for gradient estimation
            eta: Initial learning rate
            test_data: Optional test data for evaluation. If provided, displays accuracy each epoch
        """
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[x:x+mini_batch_size] for x in range(0, len(training_data), mini_batch_size)]
            for batch in mini_batches:
                self.update_mini_batch(batch, eta)
            if epoch > 0 and epoch % 20 == 0:
                eta *= 0.5
            if test_data:
                total = self.evaluate(test_data)
                print(f'Epoch {epoch}: {total}/{len(test_data)}')
            else:
                print(f'Epoch {epoch} complete')