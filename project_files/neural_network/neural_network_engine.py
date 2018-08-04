from typing import List

import numpy as np

from project_files.neural_network.network_layers import AbstractLayer, FullyConnectedLayer, WeightsHavingLayer


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
        Updates weights in layers in this network based on data from forward and backward propagation.

        :param learning_rate: value specifying how much to adjust weights in respect to gradient
        """
        for layer in self.__layer_list:
            if isinstance(layer, WeightsHavingLayer):
                layer.update_weights(learning_rate)

    @property
    def layer_list(self):
        return self.__layer_list
