"""
Module containing types of layers used in neural networks.
"""
from abc import ABC, abstractmethod

import numpy


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


class FlatteningLayer(AbstractLayer):
    """
    Layer that flattens or de-flattens data. Used to flatten data that are output of convolutional layers, so
    fully-connected layers are able to understand this.
    """

    def forward_propagation(self, input_data):
        input_image_count, channel_count, image_width, image_height = numpy.shape(input_data)
        flattened_data = numpy.reshape(input_data, (input_image_count, channel_count * image_width * image_height))
        return flattened_data

    def backward_propagation(self, input_data):
        pass


class FullyConnectedLayer(AbstractLayer):
    """
    Layer, in which every neuron from previous layer is connected to every neuron in next layer.
    """

    def forward_propagation(self, input_data):
        pass

    def backward_propagation(self, input_data):
        pass

