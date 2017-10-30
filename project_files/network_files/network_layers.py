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
    def initialize_layer(self, data_from_previous_layer):
        """
        Initializes this layer parameters based on data from previous layer.

        :param data_from_previous_layer: output data from previous layer
        :return: output data for next layer
        """
        # TODO: tu zrobic zeby inicjalizacja layerow przekazywala sobie tylko rozmiary a nie cale layery
        raise NotImplementedError

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
        image_count, channel_count, image_width, image_height = numpy.shape(input_data)
        flattened_data = numpy.reshape(input_data, (image_count, channel_count * image_width * image_height))
        return flattened_data

    def backward_propagation(self, input_data):
        pass


class FullyConnectedLayer(AbstractLayer):
    """
    Layer, in which every neuron from previous layer is connected to every neuron in next layer.
    """

    def __init__(self, output_neuron_count):
        """
        Sets the number of output neurons from this layer. To initialize theta value use `initialize_layer` method.

        :param output_neuron_count: number of output neurons from this layer
        """
        self.__output_neuron_count = output_neuron_count

    def initialize_layer(self, input_neuron_count):
        # TODO: dokonczyc ta funkcje i te ponizej
        pass

    def forward_propagation(self, input_data):
        pass

    def backward_propagation(self, input_data):
        pass

    @staticmethod
    def __add_bias(input_data):
        """
        Adds bias to given data.

        :param input_data: data to which add bias to
        :return: data wit added bias
        """
        image_count, neuron_count = numpy.shape(input_data)
        bias = numpy.ones(image_count, 1)
        data_with_bias = numpy.concatenate((bias, input_data), 1)
        return data_with_bias