"""
Module containing neural network class and builder needed to create it.
"""
from typing import List

import numpy as np

from project_files.neural_network.network_layers import AbstractLayer, FullyConnectedLayer, ConvolutionalLayer
from project_files.utils.neural_network_progress_bar import NeuralNetworkProgressBar


class NeuralNetwork:
    """
    Class used to do operations on neural network. It can do actions on it like learning and predicting learned classes.
    To create instance of this class use :class:`NeuralNetworkBuilder`.
    """

    def __init__(self, list_of_layers: List[AbstractLayer]):
        """
        Initializes empty layer list for this neural network.
        """
        self.__layer_list = list_of_layers

    def teach_network(self, input_data: np.ndarray, data_labels: np.ndarray, iteration_count: int,
                      learning_rate: float = 1):
        """
        Teaches neural network on given data.

        :param input_data: matrix of data on which network has to learn on
        :param data_labels: vector of labels of input data
        :param iteration_count: how much learning iterations the network has to execute
        :param learning_rate: value specifying how much to adjust weights in respect to gradient
        :raises: :class:`TypeError` if last layer isn't :class:`FullyConnectedLayer`
        """
        normalized_data = self.__normalize_data(input_data)
        label_matrix = self.__convert_label_vector_to_matrix(data_labels)
        progress_bar = NeuralNetworkProgressBar(iteration_count)

        for _ in progress_bar:
            data_after_forward_pass = self.__forward_propagation(normalized_data)
            error_vector = data_after_forward_pass - label_matrix
            self.__backward_propagation(error_vector)
            self.__update_weights(learning_rate)

            cost = self.__count_cost(data_after_forward_pass, label_matrix)
            progress_bar.update_cost(cost)

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Predicts output classes of input data.

        :param input_data: data to predict
        :return: vector of output classes for every data sample
        """
        normalized_data = self.__normalize_data(input_data)
        output_data = self.__forward_propagation(normalized_data)
        output_class_vector = np.argmax(output_data, axis=1)
        return output_class_vector

    def compute_numerical_gradient(self, input_data: np.ndarray, data_labels: np.ndarray) -> np.ndarray:
        """
        Computes gradient of weights in this network by counting it numerical way. This method is very slow and should
        be used only to check if gradient counted by other methods is computed correctly.

        :param input_data: data on which compute gradient
        :param data_labels: labels of input data
        :return: gradient of weights in this network
        """
        # TODO: dokonczyc

    def __convert_label_vector_to_matrix(self, label_vector: np.ndarray) -> np.ndarray:
        """
        Converts vector of values (labels) to matrix representation, so it can be easily multiplied to other matrices
        later.

        :raises: :class:`TypeError` if last layer isn't :class:`FullyConnectedLayer`
        :param label_vector: vector of labels
        :return: matrix of labels
        """
        output_neuron_count = self.__get_network_output_neuron_count()
        label_matrix = []

        for label_value in label_vector:
            row = np.zeros(output_neuron_count)
            row[label_value] = 1
            label_matrix.append(row)

        return np.array(label_matrix)

    def __get_network_output_neuron_count(self) -> int:
        """
        Gets number of neurons from last layer from this network.

        :raises: :class:`TypeError` if last layer isn't :class:`FullyConnectedLayer`
        :return: number of this network output neurons
        """
        last_layer = self.__layer_list[-1]

        if not isinstance(last_layer, FullyConnectedLayer):
            raise TypeError("Fully connected layer must be last layer")

        return last_layer.output_neuron_count

    @staticmethod
    def __normalize_data(data_to_normalize: np.ndarray) -> np.ndarray:
        """
        Normalizes given matrix - transforms values in it to range [0, 1].

        :param data_to_normalize: data to process
        :return: normalized data
        """
        # TODO: zrobic cos z tym (moze uzyc jakiejs funkcji z numpy zeby to zrobic) bo to nie normalizuje w taki sposob jak napotkalo dane uczace, tylko zawsze na podstawie aktualnych danych
        max_number = np.max(data_to_normalize)
        min_number = np.min(data_to_normalize)
        amplitude = max_number - min_number
        normalized_data = (data_to_normalize - min_number) / amplitude
        return normalized_data

    def __forward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        """
        Does forward propagation for every layer in this network based on given data.

        :param input_data: data on which to make forward pass
        :return: output data of network
        """
        data_for_next_layer = input_data

        for layer in self.__layer_list:
            data_for_next_layer = layer.forward_propagation(data_for_next_layer)

        return data_for_next_layer

    def __backward_propagation(self, input_data: np.ndarray):
        """
        Does backward propagation for every layer in this network based on given data.

        :param input_data: data that are output of neural network, used to do backward pass
        """
        data_for_previous_layer = input_data

        for layer in reversed(self.__layer_list):
            data_for_previous_layer = layer.backward_propagation(data_for_previous_layer)

    def __update_weights(self, learning_rate: float):
        """
        Updates weights in all layers in this network based on data from forward and backward propagation.

        :param learning_rate: value specifying how much to adjust weights in respect to gradient
        """
        for layer in self.__layer_list:
            layer.update_weights(learning_rate)

    @staticmethod
    def __count_cost(network_output_data: np.ndarray, data_labels: np.ndarray) -> float:
        """
        Counts cost of learned data according to formula:
            :math:`cost = - (y log(p) + (1 - y) log(1 - p))`
        where:
            * p - predicted probability of label
            * y - true value of label

        :param network_output_data: predicted data outputted by neural network
        :param data_labels: labels of data
        :return: cost of learned data
        """
        data_samples_count = np.shape(network_output_data)[0]

        # TODO: sprawdzic czy tu na pewno jest normalny iloczyn macierzy czy ten drugi
        first_component = np.transpose(data_labels) @ np.log(network_output_data)
        second_component = (1 - np.transpose(data_labels)) @ np.log(1 - network_output_data)
        cost = -(first_component + second_component) / data_samples_count
        # TODO: zobaczyc czy da sie cos zrobic z rym [0][0]
        return cost[0][0]


class NeuralNetworkBuilder:
    """
    Builder used to build neural network with given layers.
    """

    def __init__(self):
        """
        Initializes empty neural network.
        """
        self.__layer_list = []

    def add_layer(self, layer_to_add: AbstractLayer) -> "NeuralNetworkBuilder":
        """
        Adds layer to neural network.

        :param layer_to_add: layer to add to network
        :return: this builder instance, so this method can be chained
        """
        self.__layer_list.append(layer_to_add)
        return self

    def build(self, input_data_dimensions: tuple) -> NeuralNetwork:
        """
        Initializes and returns neural network with earlier provided layers.

        :raises: :class:`TypeError` when layers are in wrong order
        :return: built neural network
        """
        self.__validate_layers_order()
        self.__initialize_layers(input_data_dimensions)
        return NeuralNetwork(self.__layer_list)

    def __validate_layers_order(self):
        """
        Validates if layers are in proper order.

        :raises: :class:`TypeError` when layers are in wrong order
        """
        if not isinstance(self.__layer_list[-1], FullyConnectedLayer):
            raise TypeError("Fully connected layer must be last layer")

        for previous_layer, next_layer in zip(self.__layer_list[:-1], self.__layer_list[1:]):
            if not isinstance(previous_layer, ConvolutionalLayer) and isinstance(next_layer, ConvolutionalLayer):
                raise TypeError("Layer that's preceding ConvolutionalLayer must be another ConvolutionalLayer")

    def __initialize_layers(self, input_data_dimensions: tuple):
        """
        Initializes layers of built network.

        :param input_data_dimensions: dimensions of single input data sample
        """
        next_layer_dimensions = input_data_dimensions

        for layer in self.__layer_list:
            next_layer_dimensions = layer.initialize_layer(next_layer_dimensions)
