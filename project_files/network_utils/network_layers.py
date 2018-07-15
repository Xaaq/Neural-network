"""
Module containing types of layers used in neural networks.
"""
from abc import ABC, abstractmethod

import numpy

from project_files.network_utils.activation_functions import SigmoidFunction


class AbstractLayer(ABC):
    """
    Abstract base class for all types of layers in neural network.
    """

    @abstractmethod
    def initialize_layer(self, input_data_dimensions):
        """
        Initializes this layer parameters based on data from previous layer.

        :param input_data_dimensions: tuple of dimensions of single image data coming into this layer
        :type input_data_dimensions: tuple of int
        :return: tuple of dimensions of single output image data coming from this layer
        :rtype: tuple of int
        """
        # TODO: zrobic zeby nie trzeba bylo wywolywac tej metody, tylko zeby layery bylyinicjalizowane w konsruktorze (chociaz zobaczyc czy to czegos nie zepsuje)
        raise NotImplementedError

    @abstractmethod
    def forward_propagation(self, input_data):
        """
        Does forward pass through this layer and returns its output.

        :param input_data: data on which make forward pass
        :return: output data of this layer
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

    @abstractmethod
    def update_weights(self):
        """
        Updates weights of layer based on data gathered from forward and back propagation passes.
        """
        raise NotImplementedError


class FlatteningLayer(AbstractLayer):
    """
    Layer that flattens or de-flattens data. Used to flatten data that are output of convolutional layers, so
    fully-connected layers are able to understand this.
    """

    def __init__(self):
        """
        Creates this layer with data set to None. To initialize this layer's data use `initialize_layer` method.
        """
        self.__input_channel_count = None
        self.__input_image_width = None
        self.__input_image_height = None
        self.__output_image_neurons = None
    # TODO: zrobic __output_image_neurons jako property klasy (chociaz zobaczyc czy to czegos nie zepsuje)
    def initialize_layer(self, input_data_dimensions):
        (self.__input_channel_count,
         self.__input_image_width,
         self.__input_image_height) = input_data_dimensions

        self.__output_image_neurons = (self.__input_channel_count
                                       * self.__input_image_width
                                       * self.__input_image_height)
        return self.__output_image_neurons

    def forward_propagation(self, input_data):
        image_count, _, _, _ = numpy.shape(input_data)
        flattened_data = numpy.reshape(input_data, (image_count, self.__output_image_neurons))
        return flattened_data

    def backward_propagation(self, input_data):
        image_count, _ = numpy.shape(input_data)
        multidimensional_data = numpy.reshape(input_data, (image_count,
                                                           self.__input_channel_count,
                                                           self.__input_image_width,
                                                           self.__input_image_height))
        return multidimensional_data

    def update_weights(self):
        pass


class FullyConnectedLayer(AbstractLayer):
    """
    Layer, in which every neuron from previous layer is connected to every neuron in next layer.
    """

    # TODO: zobaczyc czy nie lepiej zrobić sub-layery jako osobne klasy
    def __init__(self, output_neuron_count):
        """
        Sets the number of output neurons from this layer. To initialize theta value use `initialize_layer` method.

        :param output_neuron_count: number of output neurons from this layer
        """
        self.__output_neuron_count = output_neuron_count
        self.__theta_matrix = None
        self.__data_before_forward_multiplication = None
        self.__data_before_backward_multiplication = None

    def initialize_layer(self, input_data_dimensions):
        self.__theta_matrix = self.__random_initialize_theta(input_data_dimensions, self.__output_neuron_count)
        return self.__output_neuron_count

    def forward_propagation(self, input_data):
        data_with_bias = self.__add_bias(input_data)
        multiplied_data = self.__multiply_by_transposed_theta(data_with_bias)
        activated_data = SigmoidFunction.calculate_result(multiplied_data)

        self.__data_before_forward_multiplication = data_with_bias
        return activated_data

    def backward_propagation(self, input_data):
        data_after_gradient = input_data * SigmoidFunction.calculate_gradient(input_data)
        multiplied_data = self.__multiply_by_theta(data_after_gradient)
        data_with_removed_bias = self.__remove_bias(multiplied_data)

        self.__data_before_backward_multiplication = data_after_gradient
        return data_with_removed_bias

    def update_weights(self):
        # TODO: dac tu alphe do parametru
        self.__theta_matrix -= 0.1 * self.__count_weights_gradient()

    @staticmethod
    def __random_initialize_theta(input_neuron_count, output_neuron_count):
        """
        Randomly initializes theta matrix based in number of input and output neurons. All values in matrix are
        initialized in range [-0.5, 0.5].

        :param input_neuron_count: number of input neurons
        :param output_neuron_count: number of output neurons
        :return: randomly initialized theta matrix
        """
        theta = numpy.random.rand(output_neuron_count, input_neuron_count + 1)
        theta -= 0.5
        return theta

    @staticmethod
    def __add_bias(input_data):
        """
        Adds bias to given data.

        :param input_data: data to which add bias to
        :return: data with added bias
        """
        image_count, _ = numpy.shape(input_data)
        bias = numpy.ones((image_count, 1))
        data_with_bias = numpy.concatenate((bias, input_data), 1)
        return data_with_bias

    @staticmethod
    def __remove_bias(input_data):
        """
        Removes bias from given data.

        :param input_data: data to remove bias from
        :return: data with removed bias
        """
        return input_data[:, 1:]

    def __multiply_by_transposed_theta(self, input_data):
        """
        Does multiplication of data by transposed theta matrix.

        :param input_data: data to multiply by transposed theta matrix
        :return: data multiplied by transposed theta matrix
        """
        transposed_theta = numpy.transpose(self.__theta_matrix)
        multiplied_data = numpy.dot(input_data, transposed_theta)
        return multiplied_data

    def __multiply_by_theta(self, input_data):
        """
        Does multiplication of data by theta matrix.

        :param input_data: data to multiply by theta matrix
        :return: data multiplied by theta matrix
        """
        multiplied_data = numpy.dot(input_data, self.__theta_matrix)
        return multiplied_data

    def __count_weights_gradient(self):
        """
        Counts and returns gradient of weights based on data saved during forward and backward propagation. After that
        cleans variables in which these data were contained.

        :return: gradient of weights of this layer
        """
        transposed_backward_data = numpy.transpose(self.__data_before_backward_multiplication)
        weights_gradient = numpy.dot(transposed_backward_data, self.__data_before_forward_multiplication)

        self.__data_before_forward_multiplication = None
        self.__data_before_backward_multiplication = None

        return weights_gradient