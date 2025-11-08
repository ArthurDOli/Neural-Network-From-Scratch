from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
from typing import List, Tuple

"""
Main script for training a neural network on the MNIST dataset

The steps:
1. Loads and normalizes the MNIST dataset
2. Prepares training and test data with appropriate formatting
3. Creates a neural network with architecture [784, 100, 10]
4. Trains the network using Stochastic Gradient Descent
"""

dataset = fetch_openml('mnist_784')

data: np.ndarray = dataset['data']
data = (data.to_numpy() - 127.5) / 127.5

target: np.ndarray = dataset['target']
target = target.to_numpy().astype(int)

data_train = data[:60000]
data_test = data[60000:]

target_train = target[:60000]
target_test = target[60000:]

def vectorize_label(x: int) -> np.ndarray:
    """
    Converts and integer label into a one-hot encoded column vector

    Args: 
        x: Integer label (0-9) representing the digir class
    """
    zeros = np.zeros(shape=(10,1))
    zeros[x] = 1
    return zeros

training_data: List[Tuple[np.ndarray, np.ndarray]] = []

for x, y in zip(data_train, target_train):
    x_format = np.reshape(x, (784, 1))
    y_format = vectorize_label(y)
    training_data.append((x_format, y_format))

test_data: List[Tuple[np.ndarray, int]] = []

for x, y in zip(data_test, target_test):
    image = np.reshape(x, (784, 1))
    test_data.append((image, y))

from network import Network

net = Network([784, 100, 10])

net.SGD(training_data, 50, 10, 0.1, test_data)