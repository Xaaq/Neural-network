"""
Module containing neural network class and builder needed to create it.
"""
from typing import List

import numpy as np

from project_files.neural_network.error_functions import CrossEntropyErrorFunction, AbstractErrorFunction
from project_files.neural_network.network_layers import AbstractLayer, FullyConnectedLayer
from project_files.utils.data_processor import DataProcessor
from project_files.utils.neural_network_progress_bar import NeuralNetworkProgressBar


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

        if not isinstance(last_layer, FullyConnectedLayer):
            raise TypeError("Last layer isn't designed to be last one")

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
        Updates weights in all layers in this network based on data from forward and backward propagation.

        :param learning_rate: value specifying how much to adjust weights in respect to gradient
        """
        for layer in self.__layer_list:
            layer.update_weights(learning_rate)

    @property
    def layer_list(self):
        return self.__layer_list


class NeuralNetwork:
    """
    Class used to do operations on neural network. It can do actions on it like learning and predicting learned classes.
    To create instance of this class use :class:`NeuralNetworkBuilder`.
    """

    def __init__(self, network_engine: NeuralNetworkEngine, error_function: AbstractErrorFunction,
                 data_processor: DataProcessor):
        """
        Initializes empty layer list for this neural network.

        :param network_engine: engine of this neural network
        :param error_function: error function used by this network
        :param data_processor: processor of data
        """
        self.__network_engine = network_engine
        self.__error_function = error_function
        self.__data_processor = data_processor

    def teach_network(self, input_data: np.ndarray, data_labels: np.ndarray, iteration_count: int,
                      learning_rate: float = 1):
        """
        Teaches neural network on given data.

        :param input_data: matrix of data on which network has to learn on
        :param data_labels: vector of labels of input data
        :param iteration_count: how much learning iterations the network has to execute
        :param learning_rate: value specifying how much to adjust weights in respect to gradient
        """
        normalized_data = self.__data_processor.normalize_data(input_data)
        label_matrix = self.__data_processor.convert_label_vector_to_matrix(
            data_labels, self.__network_engine.get_network_output_neuron_count())
        progress_bar = NeuralNetworkProgressBar(iteration_count)

        for _ in progress_bar:
            data_after_forward_pass = self.__network_engine.forward_propagation(normalized_data)
            error_vector = data_after_forward_pass - label_matrix
            self.__network_engine.backward_propagation(error_vector)
            self.__network_engine.update_weights(learning_rate)

            error = self.__error_function.count_error(data_after_forward_pass, label_matrix)
            progress_bar.update_error(error)

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Predicts output classes of input data.

        :param input_data: data to predict
        :return: vector of output classes for every data sample
        """
        normalized_data = self.__data_processor.normalize_data(input_data)
        output_data = self.__network_engine.forward_propagation(normalized_data)
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
        self.__layer_list = []
        self.__error_function = CrossEntropyErrorFunction()
        self.__data_processor = DataProcessor()

    def set_layers(self, list_of_layers_to_set: List[AbstractLayer]) -> "NeuralNetworkBuilder":
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

    def build(self, input_data_dimensions: tuple) -> NeuralNetwork:
        """
        Initializes and returns neural network with earlier provided layers.

        :return: built neural network
        """
        self.__initialize_layers(input_data_dimensions)

        network_engine = NeuralNetworkEngine(self.__layer_list)
        neural_network = NeuralNetwork(network_engine, self.__error_function, self.__data_processor)
        return neural_network

    # TODO: zrobic drugiego buildera (albo cos takiego) zeby mozna bylo budowac tez numerical gradient calculator

    def __initialize_layers(self, input_data_dimensions: tuple):
        """
        Initializes layers of built network.

        :param input_data_dimensions: dimensions of single input data sample
        """
        next_layer_dimensions = input_data_dimensions

        for layer in self.__layer_list:
            next_layer_dimensions = layer.initialize_layer(next_layer_dimensions)
