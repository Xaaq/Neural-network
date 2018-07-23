"""
Module containing cost functions used in neural networks.
"""
from abc import abstractmethod

import numpy as np


class AbstractCostFunction:
    """
    Base class for types of cost functions.
    """

    @staticmethod
    @abstractmethod
    def count_cost(predicted_label_probabilities: np.ndarray, actual_labels: np.ndarray) -> float:
        """
        Counts error of provided data.

        :param predicted_label_probabilities: predicted data labels probabilities
        :param actual_labels: actual data labels
        :return: error of learned data
        """
        raise NotImplementedError
