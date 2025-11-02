import numpy as np
from typing import TYPE_CHECKING, Union
if TYPE_CHECKING:
    from numpy.typing import NDArray

def mse(a: 'NDArray[np.float64]', y: 'NDArray[np.float64]', n: int) -> Union[float | 'NDArray[np.float64]']:
    """
    Calculates the Mean Squared Error

    The cost is calculated for the current training example or batch, normalized by the
    total number of training examples (n).

    Args:
        a: The vector of actual outputs activations from the network
        y: The vector of desired outputs (target) corresponding to the input
        n: The total number of training examples in the full dataset

    Returns:
        The scalar value of the MSE for the given inputs
    """
    return 1/(2*n)*(np.sum(np.square(y - a)))