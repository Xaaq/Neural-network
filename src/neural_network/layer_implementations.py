"""
Module containing implementations of different layer types used in neural networks.
"""
from typing import Optional, Tuple

import numpy as np

from src.neural_network.activation_functions import ActivationFunctionLike
from src.neural_network.layer_interfaces import WeightsHavingLayerLike, LastLayerLike
from src.utils.gradient_computer import GradientComputer
from src.utils.weight_utils import WeightData


class FlatteningLayer(LastLayerLike):
    """
    Layer that flattens data. Can be used to flatten data that are output of convolutional layers, so fully-connected
    layers are able to understand them.
    """

    def __init__(self):
        """
        Creates this layer.
        """
        self.__input_data_dimensions: Optional[Tuple[int, ...]] = None

    def initialize(self, input_data_dimensions: Tuple[int, ...]) -> Tuple[int, ...]:
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


class FullyConnectedLayer(WeightsHavingLayerLike, LastLayerLike):
    """
    Layer, in which every neuron from previous layer is connected to every neuron in next layer.
    """

    def __init__(self, output_neuron_count: int, activation_function: ActivationFunctionLike):
        """
        Sets the number of output neurons from this layer.

        :param output_neuron_count: number of output neurons from this layer
        :param activation_function: activation function that will be used in this layer
        """
        self.__output_neuron_count: int = output_neuron_count
        self.__activation_function: ActivationFunctionLike = activation_function
        self.__gradient_computer: GradientComputer = GradientComputer()
        self.__weight_data: Optional[WeightData] = None
        self.__do_multiply_by_gradient: bool = True
        self.__data_before_forward_activation: Optional[np.ndarray] = None

    def initialize(self, input_data_dimensions: Tuple[int, ...]) -> Tuple[int, ...]:
        input_data_shape_length = 1

        if len(input_data_dimensions) != input_data_shape_length:
            raise ValueError("Provided data dimensions shape is wrong")

        input_neuron_count = input_data_dimensions[0]
        self.__weight_data = WeightData((self.output_neuron_count, input_neuron_count + 1))
        return (self.output_neuron_count,)

    def forward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        data_with_bias = self.__add_bias(input_data)
        multiplied_data = data_with_bias @ np.transpose(self.weight_data.weights_copy)
        activated_data = self.__activation_function.calculate_result(multiplied_data)

        self.__gradient_computer.save_data_before_forward_multiplication(data_with_bias)
        self.__data_before_forward_activation = multiplied_data
        return activated_data

    def backward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        if self.__do_multiply_by_gradient:
            input_data *= self.__activation_function.calculate_gradient(self.__data_before_forward_activation)

        multiplied_data = input_data @ self.weight_data.weights_copy
        data_with_removed_bias = multiplied_data[:, 1:]

        self.__gradient_computer.save_data_before_backward_multiplication(input_data)
        return data_with_removed_bias

    def update_weights(self, learning_rate: float):
        self.weight_data.update_weights(learning_rate, self.compute_weights_gradient())

    def mark_as_let_through(self):
        self.__do_multiply_by_gradient = False

    def compute_weights_gradient(self) -> np.ndarray:
        return self.__gradient_computer.compute_weights_gradient()

    @property
    def output_neuron_count(self) -> int:
        return self.__output_neuron_count

    @property
    def weight_data(self) -> WeightData:
        return self.__weight_data

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


class ConvolutionalLayer(WeightsHavingLayerLike):
    """
    Layer which does convolution on provided data samples. It works similarly to fully connected layer, but it connects
    only chosen neurons from previous to next layer.
    """

    def initialize(self, input_data_dimensions: Tuple[int, ...]) -> Tuple[int, ...]:
        pass

    def forward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        pass

    def backward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        pass

    def update_weights(self, learning_rate: float):
        pass

    def mark_as_let_through(self):
        pass

    def compute_weights_gradient(self) -> np.ndarray:
        pass

    @property
    def weight_data(self) -> WeightData:
        pass
