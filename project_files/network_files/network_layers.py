"""
Module containing types of layers used in neural networks.
"""
from abc import ABC, abstractmethod


class AbstractLayer(ABC):
    """
    Abstract base class for all types of layers in neural network.
    """

    @abstractmethod
    def forward_propagation(self, input_data):
        # TODO: sprawdzic docstring tej funkcji i to co zwraca
        """
        Does forward pass through this layer and returns tuple of outputs: one before and one after activation function.

        :param input_data: data on which make forward pass
        :return: data before activation function, data after activation function
        :rtype: AbstractLayer.ForwardPropagationData
        """
        raise NotImplementedError

    @abstractmethod
    def backward_propagation(self, input_data):
        """
        Does backward pass through this layer and return output.

        :param input_data: data on which make backward pass
        :return: output of this layer
        """
        raise NotImplementedError
