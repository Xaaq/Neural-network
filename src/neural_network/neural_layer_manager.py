"""
Module containing manager of neural network layers.
"""
from typing import List, Callable, Type, Tuple

import numpy as np

from src.neural_network.layer_interfaces import LayerLike, WeightsHavingLayerLike, LastLayerLike
from src.utils.data_processor import Dataset


class NetworkLayerManager:
    """
    Manager of neural network layers.
    """

    def __init__(self, list_of_layers: List[LayerLike], input_data_dimensions: Tuple[int, ...]):
        """
        Initializes empty layer list for this neural network.

        :param list_of_layers: list of layers used by this network
        :param input_data_dimensions: dimensions of single data sample - used to initialize layers
        """
        self.__layer_list: List[LayerLike] = self.__initialize_layers(list_of_layers, input_data_dimensions)

    def two_way_propagation(self, dataset: Dataset) -> Dataset:
        """
        Executes forward and then backward propagation. Returns data after forward pass,

        :param dataset: dataset used to do two-way propagation
        :return: data after forward pass
        """
        data_after_forward_pass = self.forward_propagation(dataset.data)
        error_matrix = data_after_forward_pass - dataset.label_matrix
        self.backward_propagation(error_matrix)

        output_dataset = Dataset(dataset.data, data_after_forward_pass)
        return output_dataset

    def forward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        """
        Does forward propagation for every layer in this network based on given data.

        :param input_data: data on which to make forward pass
        :return: output labels in form of matrix
        """
        data_for_next_layer = input_data

        for layer in self.__layer_list:
            data_for_next_layer = layer.forward_propagation(data_for_next_layer)

        return data_for_next_layer

    def backward_propagation(self, data_error: np.ndarray):
        """
        Does backward propagation for every layer in this network based on given data.

        :param data_error: error of data that are output of neural network
        """
        data_for_previous_layer = data_error

        for layer in reversed(self.__layer_list):
            data_for_previous_layer = layer.backward_propagation(data_for_previous_layer)

    def update_weights(self, learning_rate: float):
        """
        Updates weights in layers in this network based on data from forward and backward propagation.

        :param learning_rate: value specifying how much to adjust weights in respect to gradient
        """
        for layer in self.__layer_list:
            if isinstance(layer, WeightsHavingLayerLike):
                layer.update_weights(learning_rate)

    def get_network_output_neuron_count(self) -> int:
        """
        Gets number of neurons from last layer from this network.

        :return: number of this network output neurons
        :raises TypeError: if last layer isn't designed to be last one
        """
        last_layer = self.__layer_list[-1]

        if not isinstance(last_layer, LastLayerLike):
            raise TypeError(f"Last layer must be implementing {LastLayerLike.__name__} interface")

        return last_layer.output_neuron_count

    def for_each_layer(self, function: Callable[[LayerLike], None], layer_type: Type[LayerLike] = LayerLike):
        """
        Executes given function on every layer of type specified as layer_type.

        :param function: function to execute on layers
        :param layer_type: type of layer to execute function on
        """
        for layer in self.__layer_list:
            if isinstance(layer, layer_type):
                function(layer)

    @staticmethod
    def __initialize_layers(network_layers: List[LayerLike], input_data_dimensions: Tuple[int, ...]) -> List[LayerLike]:
        """
        Initializes layers.

        :param network_layers: layers to initialize
        :param input_data_dimensions: dimensions of single input data sample
        :return initialized layers
        """
        next_layer_dimensions = input_data_dimensions

        for layer in network_layers:
            next_layer_dimensions = layer.initialize(next_layer_dimensions)

        for layer in reversed(network_layers):
            if isinstance(layer, WeightsHavingLayerLike):
                layer.mark_as_let_through()
                break

        return network_layers
