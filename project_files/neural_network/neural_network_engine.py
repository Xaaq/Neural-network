import itertools
from abc import ABC
from typing import List

import numpy as np

from project_files.neural_network.network_layers import AbstractLayer, WeightsHavingLayer, LastLayerLike


class GradientCalculatorLike(ABC):
    def calculate_gradient(self, layer_list: List[AbstractLayer], input_data: np.ndarray,
                           label_vector: np.ndarray) -> List[np.ndarray]:
        raise NotImplementedError


class NumericalGradientCalculator(GradientCalculatorLike):
    def calculate_gradient(self, layer_list: List[AbstractLayer], input_data: np.ndarray,
                           label_vector: np.ndarray) -> List[np.ndarray]:
        label_count = self.__network_engine.get_network_output_neuron_count()
        normalized_data, label_matrix = self.__data_processor.preprocess_data(input_data, label_vector, label_count)
        gradient_list = []

        for layer in self.__network_engine.layer_list:
            if isinstance(layer, WeightsHavingLayer):
                layer_gradient = self.__compute_single_layer_gradient(layer, normalized_data, label_matrix)
                gradient_list.append(layer_gradient)

        return gradient_list

    def __compute_single_layer_gradient(self, layer: WeightsHavingLayer, input_data: np.ndarray,
                                        data_labels: np.ndarray) -> np.ndarray:
        """
        Numerically computes gradient of all weights in single layer.

        :param layer: layer for which compute weights
        :param input_data: data on which to compute gradient
        :param data_labels: matrix of data labels
        :return: gradient of all weights in provided layer
        """
        epsilon = 1e-3
        weight_matrix_shape = np.shape(layer.weight_data.weights)
        gradient_matrix = np.zeros(weight_matrix_shape)

        for item_indices in itertools.product(*[range(dimension) for dimension in weight_matrix_shape]):
            error1 = self.__compute_single_weight_error(layer, item_indices, input_data, data_labels, epsilon)
            error2 = self.__compute_single_weight_error(layer, item_indices, input_data, data_labels, -epsilon)
            gradient_matrix[item_indices] = (error1 - error2) / (2 * epsilon)

        return gradient_matrix

    def __compute_single_weight_error(self, layer: WeightsHavingLayer, weight_indices: tuple, input_data: np.ndarray,
                                      data_labels: np.ndarray, epsilon: float) -> float:
        """
        Computes error of network with added epsilon value to single weight.

        :param layer: layer for which compute weight
        :param weight_indices: indices of weight for which compute gradient
        :param input_data: data on which to compute gradient
        :param data_labels: matrix of data labels
        :param epsilon: epsilon term indicating how much to add to weight before computing error function on it
        :return: error of network
        """
        weight_memento = layer.weight_data.save_weights()

        layer.weight_data[weight_indices] += epsilon
        data_after_forward_pass = self.__network_engine.forward_propagation(input_data)
        single_weight_error = self.__error_function.count_error(data_after_forward_pass, data_labels)

        layer.weight_data.restore_weights(weight_memento)
        return single_weight_error


class RegularGradientCalculator(GradientCalculatorLike):
    def calculate_gradient(self, layer_list: List[AbstractLayer], input_data: np.ndarray,
                           label_vector: np.ndarray) -> List[np.ndarray]:
        pass


class NeuralNetworkEngine:
    """
    Engine of neural network, that has all core tools needed in neural network computing.
    """

    def __init__(self, list_of_layers: List[AbstractLayer]):
        """
        Initializes empty layer list for this neural network.

        :param list_of_layers: list of layers used by this network
        """
        self.__layer_list = list_of_layers

    def get_network_output_neuron_count(self) -> int:
        """
        Gets number of neurons from last layer from this network.

        :return: number of this network output neurons
        :raises TypeError: if last layer isn't designed to be last one
        """
        last_layer = self.__layer_list[-1]

        if not isinstance(last_layer, LastLayerLike):
            raise TypeError("Last layer must be implementing {0} interface".format(LastLayerLike.__name__))

        return last_layer.output_neuron_count

    def forward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        """
        Does forward propagation for every layer in this network based on given data.

        :param input_data: data on which to make forward pass
        :return: output data of network
        """
        data_for_next_layer = input_data

        for layer in self.__layer_list:
            data_for_next_layer = layer.forward_propagation(data_for_next_layer)

        return data_for_next_layer

    def backward_propagation(self, input_data: np.ndarray):
        """
        Does backward propagation for every layer in this network based on given data.

        :param input_data: data that are output of neural network, used to do backward pass
        """
        data_for_previous_layer = input_data

        for layer in reversed(self.__layer_list):
            data_for_previous_layer = layer.backward_propagation(data_for_previous_layer)

    def update_weights(self, learning_rate: float):
        """
        Updates weights in layers in this network based on data from forward and backward propagation.

        :param learning_rate: value specifying how much to adjust weights in respect to gradient
        """
        for layer in self.__layer_list:
            if isinstance(layer, WeightsHavingLayer):
                layer.update_weights(learning_rate)

    @property
    def layer_list(self) -> List[AbstractLayer]:
        """
        Returns list of layers of this network engine.

        :return: list of layers
        """
        return self.__layer_list
