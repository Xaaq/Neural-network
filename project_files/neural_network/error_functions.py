"""
Module containing error functions used in neural networks.
"""
from abc import abstractmethod

import numpy as np


class AbstractErrorFunction:
    """
    Base class for types of error functions.
    """

    @classmethod
    @abstractmethod
    def count_error(cls, label_probabilities_matrix: np.ndarray, actual_label_matrix: np.ndarray) -> float:
        """
        Counts error of provided data.

        :param label_probabilities_matrix: predicted data labels probabilities
        :param actual_label_matrix: actual data labels
        :return: error of learned data
        """
        raise NotImplementedError


class CrossEntropyErrorFunction(AbstractErrorFunction):
    """
    Class that's implementing following error function:
        :math:`error = -(y log(p) + (1 - y) log(1 - p))`
    where:
        * p - predicted probability of label
        * y - true value of label
    """

    @classmethod
    def count_error(cls, label_probabilities_matrix: np.ndarray, actual_label_matrix: np.ndarray) -> float:
        error_sum = 0

        for label_probability_vector, actual_label_vector in zip(label_probabilities_matrix, actual_label_matrix):
            error_sum += cls.__count_single_data_sample_error(label_probability_vector, actual_label_vector)

        data_samples_count = np.shape(actual_label_matrix)[0]
        average_error = error_sum / data_samples_count
        return average_error

    @staticmethod
    def __count_single_data_sample_error(label_probabilities_vector: np.ndarray,
                                         actual_label_vector: np.ndarray) -> float:
        """
        Counts error between label probability vector and actual label vector.

        :param label_probabilities_vector: vector with labels probabilities
        :param actual_label_vector: vector with actual labels
        :return: error of single data sample
        """
        first_component = actual_label_vector @ np.transpose(np.log(label_probabilities_vector))
        second_component = (1 - actual_label_vector) @ np.transpose(np.log(1 - label_probabilities_vector))
        error = -(first_component + second_component)
        return error
