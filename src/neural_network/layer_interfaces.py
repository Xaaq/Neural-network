"""
Module containing neural network layer interfaces.
"""
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from src.utils.weight_utils import WeightData


class LayerLike(ABC):
    """
    Interface for all types of layers in neural network.
    """

    @abstractmethod
    def initialize(self, input_data_dimensions: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Initializes this layer parameters based on provided data. Also returns dimensions of data coming out of this
        layer.

        This method is called by NeuralNetworkBuilder, so when not manually creating `NeuralNetwork` there is no need to
        call it.

        :param input_data_dimensions: tuple of dimensions of data sample coming into this layer
        :return: tuple of dimensions of single data sample coming out of this layer
        :raises ValueError: if provided data dimensions can't be handled by this layer
        """
        raise NotImplementedError

    @abstractmethod
    def forward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        """
        Does forward pass through this layer and returns its output.

        :param input_data: data on which make forward pass
        :return: output data of this layer
        """
        raise NotImplementedError

    @abstractmethod
    def backward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        """
        Does backward pass through this layer and returns its output.

        :param input_data: data on which make backward pass
        :return: output of this layer
        """
        raise NotImplementedError


class WeightsHavingLayerLike(LayerLike):
    """
    Interface for layers that have weights.
    """

    @abstractmethod
    def update_weights(self, learning_rate: float):
        """
        Updates weights of layer based on data gathered from forward and back propagation passes.

        :param learning_rate: value specifying how much to adjust weights in respect to gradient
        """
        raise NotImplementedError

    @abstractmethod
    def mark_as_let_through(self):
        """
        Marks layer, so when doing backpropagation it won't multiply values by activation function gradient.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_weights_gradient(self) -> np.ndarray:
        """
        Computes this layer weight's gradient.

        :return: weight's gradient
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def weight_data(self) -> WeightData:
        """
        Getter for this layer's weight data.

        :return: this layer's weight data
        """
        raise NotImplementedError


class LastLayerLike(LayerLike):
    """
    Interface specifying that layer has possibility of being last layer of neural network.
    """

    @property
    @abstractmethod
    def output_neuron_count(self) -> int:
        """
        Returns number of output neurons from this layer.

        :return: output neurons from this layer
        """
        raise NotImplementedError
