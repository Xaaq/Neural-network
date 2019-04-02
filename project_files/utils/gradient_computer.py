"""
Module containing gradient computer used to compute gradient of weights of neural network layers.
"""
from typing import Optional

import numpy as np


class GradientComputer:
    """
    Class that can store byproduct values of forward and backward propagation of neural network. Using them, it can
    compute weights' gradient.
    """

    def __init__(self):
        """
        Initializes this data container data to empty values
        """
        self.__before_forward_multiplication: Optional[np.ndarray] = None
        self.__before_backward_multiplication: Optional[np.ndarray] = None

    def compute_weights_gradient(self) -> np.ndarray:
        """
        Computes gradient of weights based on earlier saved data.

        :return: gradient of weights of this layer
        """
        transposed_backward_data = np.transpose(self.__before_backward_multiplication)
        weight_gradient = transposed_backward_data @ self.__before_forward_multiplication

        number_of_data_samples = np.shape(self.__before_forward_multiplication)[0]
        weight_gradient /= number_of_data_samples

        return weight_gradient

    def save_data_before_forward_multiplication(self, value: np.ndarray):
        """
        Saves data before forward multiplication, so it can be used to compute gradient of weights.

        :param value: new value for data before forward multiplication
        """
        self.__before_forward_multiplication = value

    def save_data_before_backward_multiplication(self, value: np.ndarray):
        """
        Saves data before backward multiplication, so it can be used to compute gradient of weights.

        :param value: new value for data before backward multiplication
        """
        self.__before_backward_multiplication = value
