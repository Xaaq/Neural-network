"""
Module containing gradient comparator that computes gradient of neural network layers' in numerical and propagation way
and then compares them.
"""
import itertools
from math import log10
from typing import List, Tuple

import numpy as np

from project_files.neural_network.error_functions import AbstractErrorFunction
from project_files.neural_network.network_layers import WeightsHavingLayerLike
from project_files.neural_network.neural_network import NetworkLayerManager
from project_files.utils.data_processor import DataProcessor


class NetworkGradientComparator:
    """
    Class that has functionality of computing gradient numerically and by propagation. Its main usage is to compare
    these gradients to check how big is difference between them.
    """

    def __init__(self, layer_manager: NetworkLayerManager, error_function: AbstractErrorFunction,
                 data_processor: DataProcessor):
        """
        Initializes this gradient comparator components.

        :param layer_manager: manager of neural network layers
        :param error_function: error function to use
        :param data_processor: data processor to use
        """
        self.__layer_manager = layer_manager
        self.__error_function = error_function
        self.__data_processor = data_processor

    def compute_gradient_difference_magnitudes(self, input_data: np.ndarray, label_vector: np.ndarray) -> List[int]:
        """
        Computes gradient of network in two ways: numerically and by propagation. Then computes average difference of
        gradients computed in both ways and returns list of order of magnitude of every layer.

        :param input_data: input data on which to compute gradients
        :param label_vector: vector of data labels
        :return: every layer order of magnitude
        """
        label_count = self.__layer_manager.get_network_output_neuron_count()
        normalized_data, label_matrix = self.__data_processor.preprocess_data(input_data, label_vector, label_count)

        numerical_gradient_list = self.__compute_numerical_gradient(normalized_data, label_matrix)
        propagation_gradient_list = self.__compute_propagation_gradient(normalized_data, label_matrix)

        gradient_list = []

        for numerical_gradient, propagation_gradient in zip(numerical_gradient_list, propagation_gradient_list):
            gradient_difference = numerical_gradient - propagation_gradient
            gradient_magnitude = round(log10(np.average(abs(gradient_difference))))
            gradient_list.append(gradient_magnitude)

        return gradient_list

    def __compute_propagation_gradient(self, input_data: np.ndarray, label_matrix: np.ndarray) -> List[np.ndarray]:
        """
        Computes and returns gradient of weights in network based on provided data using forward and backward
        propagation.

        :param input_data: data on which compute gradient
        :param label_matrix: matrix of data labels
        :return: gradient of weights in this network
        """
        self.__layer_manager.two_way_propagation(input_data, label_matrix)

        gradient_list = []
        self.__layer_manager.for_each_layer(lambda layer: gradient_list.append(layer.compute_weights_gradient()),
                                            WeightsHavingLayerLike)
        return gradient_list

    def __compute_numerical_gradient(self, input_data: np.ndarray, label_matrix: np.ndarray) -> List[np.ndarray]:
        """
        Computes and returns gradient of weights in network based on provided data in numerical way. This method is very
        slow and should be used only to check if gradient computed by other methods is correct.

        :param input_data: data on which compute gradient
        :param label_matrix: matrix of data labels
        :return: gradient of weights in this network
        """
        gradient_list = []
        self.__layer_manager.for_each_layer(lambda layer: gradient_list.append(
            self.__compute_single_layer_gradient(layer, input_data, label_matrix)
        ), WeightsHavingLayerLike)
        return gradient_list

    def __compute_single_layer_gradient(self, layer: WeightsHavingLayerLike, input_data: np.ndarray,
                                        label_matrix: np.ndarray) -> np.ndarray:
        """
        Numerically computes gradient of all weights in single layer.

        :param layer: layer for which compute weights
        :param input_data: data on which to compute gradient
        :param label_matrix: matrix of data labels
        :return: gradient of all weights in provided layer
        """
        epsilon = 1e-3
        weight_matrix_shape = np.shape(layer.weight_data.weights_copy)
        gradient_matrix = np.zeros(weight_matrix_shape)
        weight_matrix_index_ranges = [range(dimension) for dimension in weight_matrix_shape]

        for indices in itertools.product(*weight_matrix_index_ranges):
            positive_error = self.__compute_single_weight_error(layer, indices, input_data, label_matrix, epsilon)
            negative_error = self.__compute_single_weight_error(layer, indices, input_data, label_matrix, -epsilon)
            gradient_matrix[indices] = (positive_error - negative_error) / (2 * epsilon)

        return gradient_matrix

    def __compute_single_weight_error(self, layer: WeightsHavingLayerLike, weight_indices: Tuple[int, ...],
                                      input_data: np.ndarray, label_matrix: np.ndarray, epsilon: float) -> float:
        """
        Computes error of network with added epsilon value to single weight.

        :param layer: layer for which compute weight
        :param weight_indices: indices of weight for which compute gradient
        :param input_data: data on which to compute gradient
        :param label_matrix: matrix of data labels
        :param epsilon: epsilon term indicating how much to add to weight before computing error function on it
        :return: error of network
        """
        weight_memento = layer.weight_data.save_weights()

        layer.weight_data[weight_indices] += epsilon
        data_after_forward_pass = self.__layer_manager.forward_propagation(input_data)
        single_weight_error = self.__error_function.compute_error(data_after_forward_pass, label_matrix)

        layer.weight_data.restore_weights(weight_memento)
        return single_weight_error
