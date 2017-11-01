"""
Module containing neural network class and things needed to create it - builder and director.
"""
from abc import ABC, abstractmethod

import numpy

from project_files.network_files.network_layers import FlatteningLayer, FullyConnectedLayer


class NeuralNetwork:
    """
    Class used to do operations on neural network. It can do actions on it like learning and predicting learned classes.
    To create instance of this class use :class:`NeuralNetworkDirector`.
    """

    def __init__(self):
        """
        Initializes empty layer list for this neural network.
        """
        self.__layer_list = list()

    def add_layer(self, layer_to_add):
        """
        Adds layer to this network. This method returns this object so can be chained with itself.

        :param layer_to_add: layer to add to network
        :return: self
        """
        self.__layer_list.append(layer_to_add)
        return self

    def initialize_layers(self, input_data_dimensions):
        """
        Initializes all layers in this network. This method should be called after all needed layers have been added to
        the network.

        :param input_data_dimensions: tuple of dimensions of single input image data
        """
        next_layer_dimensions = input_data_dimensions

        for layer in self.__layer_list:
            next_layer_dimensions = layer.initialize_layer(next_layer_dimensions)

    def teach_network(self, input_data, data_labels):
        """
        Teaches neural network on given data.

        :param input_data: data on which network has to learn on, format of data is multi-dimensional matrix:\n
            `number of input images x number of channels in image x width of single image x height of single image`
        :param data_labels: labels of input_data, format of this is vector of labels:\n
            `number of input images x 1`
        """
        data_for_next_layer = self.__normalize_data(input_data)

        for layer in self.__layer_list:
            data_for_next_layer = layer.forward_propagation(data_for_next_layer)

        return self.__count_cost(data_for_next_layer, data_labels)

    @staticmethod
    def __normalize_data(data_to_normalize):
        """
        Normalizes given matrix - transforms values in it to range [0, 1].

        :param data_to_normalize: data to process
        :return: normalized data
        """
        max_number = numpy.max(data_to_normalize)
        min_number = numpy.min(data_to_normalize)
        difference = max_number - min_number
        normalized_data = (data_to_normalize - min_number) / difference
        return normalized_data

    @staticmethod
    def __count_cost(network_output_data, data_labels):
        """
        Counts cost of learned data.

        :param network_output_data: predicted data outputted by neural network
        :param data_labels: labels of data
        :return: cost of learned data
        """
        data_count, _ = numpy.shape(network_output_data)

        first_component = numpy.dot(numpy.transpose(data_labels),
                                    numpy.log(network_output_data))
        second_component = numpy.dot((1 - numpy.transpose(data_labels)),
                                     numpy.log(1 - network_output_data))
        cost = - (first_component + second_component) / data_count
        return cost


class AbstractNeuralNetworkBuilder(ABC):
    """
    Abstract builder used to build :class:`NeuralNetwork` class.
    """

    @abstractmethod
    def create_neural_network(self):
        """
        Creates empty neural network.
        """
        raise NotImplementedError

    @abstractmethod
    def set_layers(self):
        """
        Sets layers of neural network.
        """
        raise NotImplementedError

    @abstractmethod
    def initialize_layers(self):
        """
        Initializes layers of neural network.
        """
        raise NotImplementedError

    @abstractmethod
    def get_result(self):
        """
        Returns built neural network.

        :return: built neural network
        """
        raise NotImplementedError


class NeuralNetworkBuilder(AbstractNeuralNetworkBuilder):
    """
    Builder used to build neural network with given number of partially and fully connected layers.
    """

    def __init__(self):
        self.__neural_network = None

    def create_neural_network(self):
        self.__neural_network = NeuralNetwork()

    def set_layers(self):
        self.__neural_network.add_layer(FlatteningLayer())
        self.__neural_network.add_layer(FullyConnectedLayer(25))
        self.__neural_network.add_layer(FullyConnectedLayer(25))
        self.__neural_network.add_layer(FullyConnectedLayer(1))

    def initialize_layers(self):
        input_data_dimensions = (1, 50, 50)
        self.__neural_network.initialize_layers(input_data_dimensions)

    def get_result(self):
        return self.__neural_network


class NeuralNetworkDirector:
    """
    Director used to create neural network. To use it, first it is needed to have neural network builder initialized.
    """

    def __init__(self, builder):
        """
        Initializes this director with given builder.

        :param builder: builder used to build this class
        :type builder: AbstractNeuralNetworkBuilder
        """
        self.__builder = builder

    def construct(self):
        """
        Constructs neural network and returns it.

        :return: constructed neural network
        :rtype: NeuralNetwork
        """
        self.__builder.create_neural_network()
        self.__builder.set_layers()
        self.__builder.initialize_layers()
        return self.__builder.get_result()
