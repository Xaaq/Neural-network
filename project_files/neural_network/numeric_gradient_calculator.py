import itertools
from abc import ABC
from typing import List

import numpy as np

from project_files.neural_network.network_layers import AbstractLayer, WeightsHavingLayer


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
        single_weight_error = self.__error_function.compute_error(data_after_forward_pass, data_labels)

        layer.weight_data.restore_weights(weight_memento)
        return single_weight_error


class RegularGradientCalculator(GradientCalculatorLike):
    def calculate_gradient(self, layer_list: List[AbstractLayer], input_data: np.ndarray,
                           label_vector: np.ndarray) -> List[np.ndarray]:
        pass
