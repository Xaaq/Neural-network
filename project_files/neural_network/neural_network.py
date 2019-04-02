"""
Module containing neural network class and builder needed to create it.
"""
from typing import List, Tuple

import numpy as np

from project_files.neural_network.error_functions import CrossEntropyErrorFunction, AbstractErrorFunction
from project_files.neural_network.network_layers import LayerLike
from project_files.neural_network.neural_layer_manager import NetworkLayerManager
from project_files.utils.data_processor import DataProcessor
from project_files.utils.neural_network_progress_bar import NeuralNetworkProgressBar


class NeuralNetwork:
    """
    Class used to do operations on neural network. It can do actions on it like learning and predicting learned classes.
    To create instance of this class use :class:`NeuralNetworkBuilder`.
    """

    def __init__(self, layer_manager: NetworkLayerManager, error_function: AbstractErrorFunction,
                 data_processor: DataProcessor):
        """
        Initializes this neural network components.

        :param layer_manager: manager of neural network layers
        :param error_function: error function used by this network
        :param data_processor: processor of data
        """
        self.__layer_manager = layer_manager
        self.__error_function = error_function
        self.__data_processor = data_processor

    def fit(self, input_data: np.ndarray, label_vector: np.ndarray, iteration_count: int, learning_rate: float = 1):
        """
        Fit this network to given data.

        :param input_data: matrix of data on which network has to learn on
        :param label_vector: vector of labels of input data
        :param iteration_count: how much learning iterations the network has to execute
        :param learning_rate: value specifying how much to adjust weights in respect to gradient
        """
        label_count = self.__layer_manager.get_network_output_neuron_count()
        normalized_data, label_matrix = self.__data_processor.preprocess_data(input_data, label_vector, label_count)
        progress_bar = NeuralNetworkProgressBar(iteration_count)

        for _ in progress_bar:
            data_after_forward_pass = self.__layer_manager.forward_propagation(normalized_data)
            error_matrix = data_after_forward_pass - label_matrix
            self.__layer_manager.backward_propagation(error_matrix)
            self.__layer_manager.update_weights(learning_rate)

            error = self.__error_function.compute_error(data_after_forward_pass, label_matrix)
            progress_bar.update_error(error)

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Predicts output classes of input data.

        :param input_data: data to predict
        :return: vector of output classes for every data sample
        """
        normalized_data = self.__data_processor.normalize_data(input_data)
        output_data = self.__layer_manager.forward_propagation(normalized_data)
        output_class_vector = self.__data_processor.convert_label_matrix_to_vector(output_data)
        return output_class_vector


class NeuralNetworkBuilder:
    """
    Builder used to build neural network with given layers.
    """

    def __init__(self):
        """
        Initializes parameters used to build neural network.
        """
        self.__layer_list: List[LayerLike] = []
        self.__error_function = CrossEntropyErrorFunction()
        self.__data_processor = DataProcessor()

    def set_layers(self, list_of_layers_to_set: List[LayerLike]) -> "NeuralNetworkBuilder":
        """
        Sets network layers to given ones.

        :param list_of_layers_to_set: list of layers to set for network
        :return: this builder instance
        """
        self.__layer_list = list_of_layers_to_set
        return self

    def set_error_function(self, error_function: AbstractErrorFunction) -> "NeuralNetworkBuilder":
        """
        Sets error function used in neural network.

        :param error_function: error function to use
        :return: this builder instance
        """
        self.__error_function = error_function
        return self

    def set_data_processor(self, data_processor: DataProcessor) -> "NeuralNetworkBuilder":
        """
        Sets data processor of network.

        :param data_processor: data processor to use
        :return: this builder instance
        """
        self.__data_processor = data_processor
        return self

    def build(self, input_data_dimensions: Tuple[int]) -> NeuralNetwork:
        """
        Initializes and returns neural network with earlier provided layers.

        :param input_data_dimensions: dimensions of single data sample
        :return: built neural network
        """
        layer_manager = NetworkLayerManager(self.__layer_list, input_data_dimensions)
        neural_network = NeuralNetwork(layer_manager, self.__error_function, self.__data_processor)
        return neural_network
