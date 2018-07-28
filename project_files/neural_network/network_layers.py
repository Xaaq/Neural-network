"""
Module containing types of layers used in neural networks.
"""
from abc import ABC, abstractmethod
from typing import Type

import numpy as np

from project_files.neural_network.activation_functions import SigmoidFunction, AbstractActivationFunction


class AbstractLayer(ABC):
    """
    Abstract base class for all types of layers in neural network.
    """

    @abstractmethod
    def initialize_layer(self, input_data_dimensions: tuple) -> tuple:
        """
        Initializes this layer parameters based on provided data. Also returns dimensions of data coming out of this
        layer.

        :param input_data_dimensions: tuple of dimensions of data sample coming into this layer
        :return: tuple of dimensions of single data sample coming out of this layer
        """
        raise NotImplementedError

    @abstractmethod
    def forward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        """
        Does forward pass through this layer and returns its output.

        :param input_data: data on which make forward pass
        :return: output data of this layer
        """
        raise NotImplementedError

    @abstractmethod
    def backward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        """
        Does backward pass through this layer and returns its output.

        :param input_data: data on which make backward pass
        :return: output of this layer
        """
        raise NotImplementedError

    @abstractmethod
    def update_weights(self, learning_rate: float):
        """
        Updates weights of layer based on data gathered from forward and back propagation passes.

        :param learning_rate: value specifying how much to adjust weights in respect to gradient
        """
        raise NotImplementedError


class FlatteningLayer(AbstractLayer):
    """
    Layer that flattens data. Can be used to flatten data that are output of convolutional layers, so fully-connected
    layers are able to understand them.
    """

    def __init__(self):
        """
        Creates this layer.
        """
        self.__input_data_dimensions = None

    def initialize_layer(self, input_data_dimensions: tuple) -> tuple:
        self.__input_data_dimensions = input_data_dimensions
        return (self.__output_neuron_count,)

    def forward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        data_samples_count = np.shape(input_data)[0]
        flattened_data = np.reshape(input_data, (data_samples_count, self.__output_neuron_count))
        return flattened_data

    def backward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        data_samples_count = np.shape(input_data)[0]
        multidimensional_data = np.reshape(input_data, (data_samples_count, *self.__input_data_dimensions))
        return multidimensional_data

    def update_weights(self, learning_rate: float):
        pass

    @property
    def __output_neuron_count(self) -> int:
        """
        Counts output neuron count of this layer based on number of input data dimensions.

        :return: number of neurons coming out of this layer
        """
        output_neuron_count = 1

        for dimension in self.__input_data_dimensions:
            output_neuron_count *= dimension

        return output_neuron_count


class FullyConnectedLayer(AbstractLayer):
    """
    Layer, in which every neuron from previous layer is connected to every neuron in next layer.
    """
    __input_data_shape_length = 1

    def __init__(self, output_neuron_count: int,
                 activation_function: Type[AbstractActivationFunction] = SigmoidFunction, is_last_layer=False):
        """
        Sets the number of output neurons from this layer.

        :param output_neuron_count: number of output neurons from this layer
        """
        self.__output_neuron_count = output_neuron_count
        self.__activation_function = activation_function
        self.__weight_matrix = None
        self.__data_before_forward_activation = None
        self.__data_before_backward_multiplication = None
        self.__is_last_layer = is_last_layer

    def initialize_layer(self, input_data_dimensions: tuple) -> tuple:
        if len(input_data_dimensions) != self.__input_data_shape_length:
            raise ValueError("Provided data dimensions shape is wrong")

        input_neuron_count = input_data_dimensions[0]
        self.__weight_matrix = self.__generate_random_weight_matrix(input_neuron_count, self.output_neuron_count)
        self.weight_matrix_copy = self.__weight_matrix.copy()
        return (self.output_neuron_count,)

    def forward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        data_with_bias = self.__add_bias(input_data)
        multiplied_data = self.__multiply_by_transposed_weights(data_with_bias)
        activated_data = self.__activation_function.calculate_result(multiplied_data)

        self.__data_after_forward_bias = data_with_bias
        self.__data_before_forward_activation = multiplied_data
        return activated_data

    def backward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        if self.__is_last_layer:
            data_after_gradient = input_data
        else:
            data_after_gradient = input_data * self.__activation_function.calculate_gradient(
                self.__data_before_forward_activation)

        self.__delta_term = data_after_gradient
        multiplied_data = self.__multiply_by_weights(data_after_gradient)
        data_with_removed_bias = self.__remove_bias(multiplied_data)
        return data_with_removed_bias

    def update_weights(self, learning_rate: float):
        self.__weight_matrix -= learning_rate * self.__count_weights_gradient()

    @property
    def output_neuron_count(self) -> int:
        """
        Returns number of output neurons from this layer.

        :return: output neurons from this layer
        """
        return self.__output_neuron_count

    @staticmethod
    def __generate_random_weight_matrix(input_neuron_count: int, output_neuron_count: int) -> np.ndarray:
        """
        Randomly initializes weight matrix based on number of input and output neurons. All values in matrix are
        initialized in range [-0.5, 0.5].

        :param input_neuron_count: number of input neurons
        :param output_neuron_count: number of output neurons
        :return: randomly initialized weight matrix
        """
        weights = np.random.rand(output_neuron_count, input_neuron_count + 1) - 0.5
        return weights

    @staticmethod
    def __add_bias(input_data: np.ndarray) -> np.ndarray:
        """
        Adds bias term to given data.

        :param input_data: data to which add bias term to
        :return: data with added bias term
        """
        data_samples_count = np.shape(input_data)[0]
        bias = np.ones((data_samples_count, 1))
        data_with_bias = np.concatenate((bias, input_data), 1)
        return data_with_bias

    @staticmethod
    def __remove_bias(input_data: np.ndarray) -> np.ndarray:
        """
        Removes bias term from given data.

        :param input_data: data to remove bias term from
        :return: data with removed bias term
        """
        return input_data[:, 1:]

    def __multiply_by_transposed_weights(self, input_data: np.ndarray) -> np.ndarray:
        """
        Does multiplication of data by transposed weight matrix.

        :param input_data: data to multiply by transposed weight matrix
        :return: data multiplied by transposed weight matrix
        """
        transposed_weights = np.transpose(self.__weight_matrix)
        multiplied_data = input_data @ transposed_weights
        return multiplied_data

    def __multiply_by_weights(self, input_data: np.ndarray) -> np.ndarray:
        """
        Does multiplication of data by weight matrix.

        :param input_data: data to multiply by weight matrix
        :return: data multiplied by weight matrix
        """
        multiplied_data = input_data @ self.__weight_matrix
        return multiplied_data

    def __count_weights_gradient(self) -> np.ndarray:
        """
        Counts and returns gradient of weights based on data saved during forward and backward propagation.

        :return: gradient of weights of this layer
        """
        transposed_backward_data = np.transpose(self.__delta_term)
        number_of_examples = np.shape(self.__data_after_forward_bias)[0]
        weights_gradient = transposed_backward_data @ self.__data_after_forward_bias
        weights_gradient /= number_of_examples

        return weights_gradient


class ConvolutionalLayer(AbstractLayer):
    """
    Layer which does convolution on provided data samples. It works similarly to fully connected layer, but it connects
    only chosen neurons from previous to next layer.
    """

    def initialize_layer(self, input_data_dimensions: tuple) -> tuple:
        pass

    def forward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        pass

    def backward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        pass

    def update_weights(self, learning_rate: float):
        pass
