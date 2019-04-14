"""
Module containing activation functions used in neural networks.
"""
from abc import ABC, abstractmethod

import numpy as np


class ActivationFunctionLike(ABC):
    """
    Base class for activation functions used in neural networks.
    """

    @staticmethod
    @abstractmethod
    def calculate_result(input_data: np.ndarray) -> np.ndarray:
        """
        Does calculations on input data and returns result.

        :param input_data: input data to calculate function on
        :return: result of function
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def calculate_gradient(cls, input_data: np.ndarray) -> np.ndarray:
        """
        Calculates gradient of function on input and returns result.

        :param input_data: input data to calculate gradient on
        :return: result of gradient of function
        """
        raise NotImplementedError


class ReluFunction(ActivationFunctionLike):
    """
    RELU function with following formula:
        `if x < 0 then y = 0`

        `if x >= 0 then y = x`
    """

    @staticmethod
    def calculate_result(input_data: np.ndarray) -> np.ndarray:
        output_data = (input_data > 0) * input_data
        return output_data

    @classmethod
    def calculate_gradient(cls, input_data: np.ndarray) -> np.ndarray:
        output_data = (input_data > 0)
        return output_data


class SigmoidFunction(ActivationFunctionLike):
    """
    Sigmoid function with following formula:
        :math:`y = 1 / (1 + e^{-x})`
    """

    @staticmethod
    def calculate_result(input_data: np.ndarray) -> np.ndarray:
        output_data = 1 / (1 + np.exp(-input_data))
        return output_data

    @classmethod
    def calculate_gradient(cls, input_data: np.ndarray) -> np.ndarray:
        function_value = cls.calculate_result(input_data)
        output_data = function_value * (1 - function_value)
        return output_data
