"""
Module containing neural network class and things needed to create it - builder and director.
"""
from abc import ABC, abstractmethod

import numpy


class NeuralNetwork:
    """
    Class used to contain neural network. It can do actions on it like learning and predicting learned classes.
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
        pass

    def get_result(self):
        pass

    def set_layers(self):
        pass


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
        self.__builder.set_layers()
        return self.__builder.get_result()
