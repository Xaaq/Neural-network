"""
Module containing utilities related to neural network layer's weights.
"""
import numpy as np

from project_files.utils.gradient_calculator import GradientCalculator


class WeightMemento:
    """
    Memento used to remember neural network layer's weights.
    """

    def __init__(self, weights: np.ndarray):
        """
        Memorizes provided weights.

        :param weights: weights to memorize
        """
        self.__weights = weights.copy()

    def get_weights(self) -> np.ndarray:
        """
        Returns earlier memorized weights.

        :return: memorized weights
        """
        return self.__weights.copy()


class WeightData:
    """
    Data object that encapsulates layer weights and allows to do computing on them.
    """

    def __init__(self, weight_dimensions: tuple):
        """
        Initializes this data container weights with provided dimensions.

        :param weight_dimensions: dimensions of weights that will be generated
        """
        self.__weights = self.__generate_random_weight_matrix(weight_dimensions)

    def __getitem__(self, indices: tuple) -> float:
        """
        Gets single weight from given indices.

        :param indices: indices of weight to get
        :return: single weight
        """
        return self.weights[indices]

    def __setitem__(self, indices: tuple, value: float):
        """
        Sets particular weight to given value.

        :param indices: indices of weights element to set
        :param value: value to set
        """
        self.__weights[indices] = value

    def update_weights(self, learning_rate: float, gradient_calculator: GradientCalculator):
        """
        Updates weights using provided data.

        :param learning_rate: multiplier of weights update
        :param gradient_calculator: gradient calculator needed to compute weight gradient
        """
        self.__weights -= learning_rate * gradient_calculator.count_weight_gradient()

    def save_weights(self) -> WeightMemento:
        """
        Saves weights in memento object.

        :return: memento object with saved weights
        """
        return WeightMemento(self.weights)

    def restore_weights(self, memento: WeightMemento):
        """
        Restores weights from memento object.

        :param memento: memento object from which to restore weight data
        """
        self.__weights = memento.get_weights()

    @property
    def weights(self) -> np.ndarray:
        """
        Returns copy of stored weights.

        :return: copy of stored weights
        """
        return self.__weights.copy()

    @staticmethod
    def __generate_random_weight_matrix(weight_dimensions: tuple) -> np.ndarray:
        """
        Randomly initializes weight matrix based on provided dimensions. All values in matrix are initialized in range
        [-0.5, 0.5].

        :param weight_dimensions: dimensions of weights to create
        :return: randomly initialized weight matrix
        """
        weights = np.random.rand(*weight_dimensions) - 0.5
        return weights
