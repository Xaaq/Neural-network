"""
Module containing activation functions used in neural networks.
"""
from abc import ABC, abstractmethod

import numpy as np


# TODO: sprawdzic czy te funkcje (i ogolnie wsyzstkie inne jakie zrobilem) moga przyjmowac wielowymiarowe wektory i dobrze je zwracaja
class AbstractActivationFunction(ABC):
    """
    Base class for activation functions used in neural networks.
    """

    @staticmethod
    @abstractmethod
    def calculate_result(input_data: np.ndarray) -> np.ndarray:
        """
        Does calculations on input and returns result.

        :param input_data: input data to calculate function on
        :return: result of function
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def calculate_gradient(input_data: np.ndarray) -> np.ndarray:
        """
        Calculates gradient of function on input and returns result.

        :param input_data: input data to calculate gradient on
        :return: result of gradient of function
        """
        raise NotImplementedError


class ReluFunction(AbstractActivationFunction):
    """
    RELU function used in neural networks. It works as follows:
        `if x < 0 then y = 0`\n
        `if x >= 0 then y = x`
    """

    @staticmethod
    def calculate_result(input_data: np.ndarray) -> np.ndarray:
        output_data = (input_data > 0) * input_data
        return output_data

    @staticmethod
    def calculate_gradient(input_data: np.ndarray) -> np.ndarray:
        output_data = (input_data > 0)
        return output_data


class SigmoidFunction(AbstractActivationFunction):
    """
    Sigmoid function used in neural networks. Equation of this function is as follows:
        `y = 1 / (1 + e^(-x))`
    """

    @staticmethod
    def calculate_result(input_data: np.ndarray) -> np.ndarray:
        output_data = 1 / (1 + np.exp(-input_data))
        return output_data

    @staticmethod
    def calculate_gradient(input_data: np.ndarray) -> np.ndarray:
        function_value = SigmoidFunction.calculate_result(input_data)
        output_data = function_value * (1 - function_value)
        return output_data
