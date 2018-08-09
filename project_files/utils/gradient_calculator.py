"""
Module containing gradient calculator used to compute gradient of weights of neural network layers.
"""
import numpy as np


class GradientCalculator:
    """
    Data object that can store byproduct values of forward and backward propagation of neural network. Using them, it
    can compute weight gradient.
    """

    def __init__(self):
        """
        Initializes this data container data to empty values
        """
        self.__before_forward_multiplication: np.ndarray = None
        self.__before_backward_multiplication: np.ndarray = None

    def count_weight_gradient(self) -> np.ndarray:
        """
        Counts and returns gradient of weights based on earlier saved data.

        :return: gradient of weights of this layer
        """
        transposed_backward_data = np.transpose(self.before_backward_multiplication)
        weight_gradient = transposed_backward_data @ self.before_forward_multiplication

        number_of_data_samples = np.shape(self.before_forward_multiplication)[0]
        weight_gradient /= number_of_data_samples

        return weight_gradient

    # TODO: usunac gettery
    @property
    def before_forward_multiplication(self) -> np.ndarray:
        """
        Getter for data before forward multiplication.

        :return: data before forward multiplication
        """
        return self.__before_forward_multiplication

    @before_forward_multiplication.setter
    def before_forward_multiplication(self, value: np.ndarray):
        """
        Setter for data before forward multiplication.

        :param value: new value for data before forward multiplication
        """
        self.__before_forward_multiplication = value

    @property
    def before_backward_multiplication(self) -> np.ndarray:
        """
        Getter for data before backward multiplication.

        :return: data before backward multiplication
        """
        return self.__before_backward_multiplication

    @before_backward_multiplication.setter
    def before_backward_multiplication(self, value: np.ndarray):
        """
        Setter for data before backward multiplication.

        :param value: new value for data before backward multiplication
        """
        self.__before_backward_multiplication = value
