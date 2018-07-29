"""
Module containing memento classes.
"""
import numpy as np


class WeightMemento:
    """
    Memento used to remember weights values in single neural network layer.
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
