"""
Module containing activation functions used in neural networks.
"""
from abc import ABC, abstractmethod

import numpy


class AbstractActivationFunction(ABC):
    """
    Base class for activation functions used in neural networks.
    """

    @staticmethod
    @abstractmethod
    def calculate_result(input_data):
        """
        Does calculations on input and returns result.

        :param input_data: input data to calculate function on
        :return: result of function
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def calculate_gradient(input_data):
        """
        Calculates gradient of function on input and returns result.

        :param input_data: input data to calculate gradient on
        :return: result of gradient of function
        """
        raise NotImplementedError


class ReluFunction(AbstractActivationFunction):
    """
    RELU function used in neural networks. It works as follows:
        `if x < 0 then y = 0`

        `if x >= 0 then y = x`
    """

    @staticmethod
    def calculate_result(input_data):
        output_data = (input_data > 0) * input_data
        return output_data

    @staticmethod
    def calculate_gradient(input_data):
        output_data = (input_data > 0)
        return output_data


class SigmoidFunction(AbstractActivationFunction):
    """
    Sigmoid function used in neural networks. Equation of this function is as follows:
        `y = 1 / (1 + e^(-x))`
    """

    @staticmethod
    def calculate_result(input_data):
        output_data = 1 / (1 + numpy.exp(-input_data))
        return output_data

    @staticmethod
    def calculate_gradient(input_data):
        function_value = SigmoidFunction.calculate_result(input_data)
        output_data = function_value * (1 - function_value)
        return output_data
