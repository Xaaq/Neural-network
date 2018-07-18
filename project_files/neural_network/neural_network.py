"""
Module containing neural network class and builder needed to create it.
"""
from typing import List

import numpy as np

from project_files.neural_network.network_layers import AbstractLayer
from project_files.utils.neural_network_progress_bar import NeuralNetworkProgressBar


# TODO: zrobic testy
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

    # TODO: zrobic zeby nie trzeba bylo podawac labelek do uczenia w osobnych mini-tablicach
    # TODO: zobaczyc czy alphe dac jako arguent tej metody czy jako jakas zmienna tej klasy
    def teach_network(self, input_data: np.ndarray, data_labels: np.ndarray):
        """
        Teaches neural network on given data.

        :param input_data: data on which network has to learn on
        :param data_labels: labels of input data
        """
        # TODO: dodac do docstring returna (albo i nie) i dodac parametr z iloscia iteracji uczenia
        normalized_data = self.__normalize_data(input_data)
        progress_bar = NeuralNetworkProgressBar(500)

        for _ in progress_bar:
            data_after_forward_pass = self.__forward_propagation(normalized_data)
            error_vector = data_after_forward_pass - data_labels
            self.__backward_propagation(error_vector)
            self.__update_weights()

            cost = self.__count_cost(data_after_forward_pass, data_labels)
            progress_bar.update_cost(cost)

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Predicts output classes of input data.

        :param input_data: data to predict
        :return: output classes for every data sample
        """
        normalized_data = self.__normalize_data(input_data)
        output_data = self.__forward_propagation(normalized_data)
        rounded_output_data = np.round(output_data)
        return rounded_output_data

    def compute_numerical_gradient(self, input_data: np.ndarray, data_labels: np.ndarray) -> np.ndarray:
        """
        Computes gradient of weights in this network by counting it numerical way. This method is very slow and should
        be used only to check if gradient counted by other methods is computed correctly.

        :param input_data: data on which compute gradient
        :param data_labels: labels of input data
        :return: gradient of weights in this network
        """
        # TODO: dokonczyc

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

    def __update_weights(self):
        """
        Updates weights in all layers in this network based on data from forward and backward propagation.
        """
        for layer in self.__layer_list:
            layer.update_weights()

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

        :return: built neural network
        """
        self.__initialize_layers(input_data_dimensions)
        return NeuralNetwork(self.__layer_list)

    def __initialize_layers(self, input_data_dimensions: tuple):
        """
        Initializes layers of built network.

        :param input_data_dimensions: dimensions of single input data sample
        """
        next_layer_dimensions = input_data_dimensions

        for layer in self.__layer_list:
            next_layer_dimensions = layer.initialize_layer(next_layer_dimensions)
