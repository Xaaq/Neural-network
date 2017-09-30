"""
Module containing neural network class and builder needed to create it.
"""
from typing import List, Tuple

from src.network_core_tools.neural_layer_manager import NetworkLayerManager
from src.network_functions.error_functions import CrossEntropyErrorFunction, ErrorFunctionLike
from src.layer_tools.layer_interfaces import LayerLike
from src.data_processing.dataset import Dataset
from src.utils.neural_network_progress_bar import NeuralNetworkProgressBar


class NeuralNetwork:
    """
    Class used to do operations on neural network. It can do actions on it like learning and predicting learned classes.
    To create instance of this class use :class:`NeuralNetworkBuilder`.
    """

    def __init__(self, layer_manager: NetworkLayerManager, error_function: ErrorFunctionLike):
        """
        Initializes this neural network components.

        :param layer_manager: manager of neural network layers
        :param error_function: error function used by this network
        """
        self.__layer_manager: NetworkLayerManager = layer_manager
        self.__error_function: ErrorFunctionLike = error_function

    def fit(self, dataset: Dataset, iteration_count: int, learning_rate: float = 1):
        """
        Fit this network to given data.

        :param dataset: dataset on which to execute learning
        :param iteration_count: how much learning iterations the network has to execute
        :param learning_rate: value specifying how much to adjust weights in respect to gradient
        """
        progress_bar = NeuralNetworkProgressBar(iteration_count)

        for _ in progress_bar:
            data_after_forward_pass = self.__layer_manager.two_way_propagation(dataset)
            self.__layer_manager.update_weights(learning_rate)

            error = self.__error_function.compute_error(data_after_forward_pass.label_matrix, dataset.label_matrix)
            progress_bar.update_error(error)

    def predict(self, dataset: Dataset) -> Dataset:
        """
        Predicts output classes of input data.

        :param dataset: dataset with filled input data
        :return: dataset with filled input data and data labels
        """
        predicted_labels = self.__layer_manager.forward_propagation(dataset.data)
        output_dataset = Dataset(dataset.data, predicted_labels)
        return output_dataset


class NeuralNetworkBuilder:
    """
    Builder used to build neural network with given parameters.
    """

    def __init__(self):
        """
        Initializes parameters used to build neural network.
        """
        self.__layer_list: List[LayerLike] = []
        self.__error_function = CrossEntropyErrorFunction()

    def set_layers(self, list_of_layers_to_set: List[LayerLike]) -> "NeuralNetworkBuilder":
        """
        Sets network layers to given ones.

        :param list_of_layers_to_set: list of layers to set for network
        :return: this builder instance
        """
        self.__layer_list = list_of_layers_to_set
        return self

    def set_error_function(self, error_function: ErrorFunctionLike) -> "NeuralNetworkBuilder":
        """
        Sets error function used in neural network.

        :param error_function: error function to use
        :return: this builder instance
        """
        self.__error_function = error_function
        return self

    def build(self, input_data_dimensions: Tuple[int, ...]) -> NeuralNetwork:
        """
        Initializes and returns neural network with earlier provided layers.

        :param input_data_dimensions: dimensions of single data sample
        :return: built neural network
        """
        layer_manager = NetworkLayerManager(self.__layer_list, input_data_dimensions)
        neural_network = NeuralNetwork(layer_manager, self.__error_function)
        return neural_network
