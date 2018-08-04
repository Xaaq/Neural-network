"""
Module containing types of layers used in neural networks.
"""
from abc import ABC, abstractmethod

import numpy as np

from project_files.neural_network.activation_functions import AbstractActivationFunction
from project_files.utils.weight_utils import WeightData, GradientCalculator


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

    @property
    @abstractmethod
    def output_neuron_count(self) -> int:
        """
        Returns number of output neurons from this layer.

        :return: output neurons from this layer
        """
        raise NotImplementedError


class WeightsHavingLayer(AbstractLayer):
    """
    Abstract base class for layers that have weights.
    """

    @abstractmethod
    def update_weights(self, learning_rate: float):
        """
        Updates weights of layer based on data gathered from forward and back propagation passes.

        :param learning_rate: value specifying how much to adjust weights in respect to gradient
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def weight_data(self) -> WeightData:
        """
        Getter for this layer's weight data.

        :return: this layer's weight data
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def gradient_calculator(self) -> GradientCalculator:
        """
        Getter for this layer's gradient calculator.

        :return: this layer's gradient calculator
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
        return (self.output_neuron_count,)

    def forward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        data_samples_count = np.shape(input_data)[0]
        flattened_data = np.reshape(input_data, (data_samples_count, self.output_neuron_count))
        return flattened_data

    def backward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        data_samples_count = np.shape(input_data)[0]
        multidimensional_data = np.reshape(input_data, (data_samples_count, *self.__input_data_dimensions))
        return multidimensional_data

    @property
    def output_neuron_count(self) -> int:
        output_neuron_count = 1

        for dimension in self.__input_data_dimensions:
            output_neuron_count *= dimension

        return output_neuron_count


class FullyConnectedLayer(WeightsHavingLayer):
    """
    Layer, in which every neuron from previous layer is connected to every neuron in next layer.
    """
    __input_data_shape_length = 1

    def __init__(self, output_neuron_count: int,
                 activation_function: AbstractActivationFunction, is_last_layer=False):
        """
        Sets the number of output neurons from this layer.

        :param output_neuron_count: number of output neurons from this layer
        :param activation_function: activation function that will be used in this layer
        :param is_last_layer: flag indicating if this is last layer of network
        """
        self.__output_neuron_count = output_neuron_count
        self.__activation_function = activation_function
        self.__is_last_layer = is_last_layer
        self.__gradient_calculator = GradientCalculator()
        self.__weight_data: WeightData = None
        self.__data_before_forward_activation: np.ndarray = None

    def initialize_layer(self, input_data_dimensions: tuple) -> tuple:
        if len(input_data_dimensions) != self.__input_data_shape_length:
            raise ValueError("Provided data dimensions shape is wrong")

        input_neuron_count = input_data_dimensions[0]
        self.__weight_data = WeightData((self.output_neuron_count, input_neuron_count + 1))
        return (self.output_neuron_count,)

    def forward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        data_with_bias = self.__add_bias(input_data)
        multiplied_data = self.__multiply_by_transposed_weights(data_with_bias)
        activated_data = self.__activation_function.calculate_result(multiplied_data)

        self.__gradient_calculator.before_forward_multiplication = data_with_bias
        self.__data_before_forward_activation = multiplied_data
        return activated_data

    def backward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        if self.__is_last_layer:
            data_after_gradient = input_data
        else:
            data_after_gradient = input_data * self.__activation_function.calculate_gradient(
                self.__data_before_forward_activation)

        self.__gradient_calculator.before_backward_multiplication = data_after_gradient
        multiplied_data = self.__multiply_by_weights(data_after_gradient)
        data_with_removed_bias = self.__remove_bias(multiplied_data)
        return data_with_removed_bias

    def update_weights(self, learning_rate: float):
        self.weight_data.update_weights(learning_rate, self.__gradient_calculator)

    @property
    def output_neuron_count(self) -> int:
        return self.__output_neuron_count

    @property
    def weight_data(self) -> WeightData:
        return self.__weight_data

    @property
    def gradient_calculator(self) -> GradientCalculator:
        return self.__gradient_calculator

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
        transposed_weights = np.transpose(self.weight_data.weights)
        multiplied_data = input_data @ transposed_weights
        return multiplied_data

    def __multiply_by_weights(self, input_data: np.ndarray) -> np.ndarray:
        """
        Does multiplication of data by weight matrix.

        :param input_data: data to multiply by weight matrix
        :return: data multiplied by weight matrix
        """
        multiplied_data = input_data @ self.weight_data.weights
        return multiplied_data


class ConvolutionalLayer(WeightsHavingLayer):
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

    @property
    def weight_data(self) -> WeightData:
        pass

    @property
    def gradient_calculator(self) -> GradientCalculator:
        pass

    @property
    def output_neuron_count(self) -> int:
        pass
