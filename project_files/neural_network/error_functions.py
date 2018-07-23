"""
Module containing error functions used in neural networks.
"""
from abc import abstractmethod
from typing import Tuple

import numpy as np


class AbstractErrorFunction:
    """
    Base class for types of error functions.
    """

    @classmethod
    @abstractmethod
    def count_error(cls, label_probabilities: np.ndarray, actual_labels: np.ndarray) -> float:
        """
        Counts error of provided data.

        :param label_probabilities: predicted data labels probabilities
        :param actual_labels: actual data labels
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
    def count_error(cls, label_probabilities: np.ndarray, actual_labels: np.ndarray) -> float:
        log_label_probabilities, inv_log_label_probabilities = cls.__prepare_logarithmized_data(label_probabilities)
        mean_error = cls.__count_mean_error(actual_labels, log_label_probabilities, inv_log_label_probabilities)
        return mean_error

    @staticmethod
    def __prepare_logarithmized_data(data_to_logarithmize: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes `log(x)` and `(1 - log(x))` functions on given data and outputs them separately.

        :param data_to_logarithmize: data to logarithmize
        :return: `log(x)` data, `(1 - log(x))` data
        """
        logarithmized_data = np.log(data_to_logarithmize)
        inverse_logarithmized_data = np.log(1 - data_to_logarithmize)
        return logarithmized_data, inverse_logarithmized_data

    @classmethod
    def __count_mean_error(cls, actual_labels: np.ndarray, log_label_probabilities: np.ndarray,
                           inv_log_label_probabilities: np.ndarray) -> float:
        """
        Counts mean error between earlier prepared logarithmized label probabilities and actual labels.

        :param actual_labels: true data labels
        :param log_label_probabilities: label probabilities with `log(x)` function used on them
        :param inv_log_label_probabilities: label probabilities with `(1 - log(x))` function used on them
        :return: mean error over all samples
        """
        error_sum = 0

        for single_label, single_log_probability, single_inv_log_probability \
                in zip(actual_labels, log_label_probabilities, inv_log_label_probabilities):
            error_sum += cls.__count_single_data_sample_error(single_label, single_log_probability,
                                                              single_inv_log_probability)

        data_samples_count = np.shape(actual_labels)[0]
        average_error = error_sum / data_samples_count
        return average_error

    @staticmethod
    def __count_single_data_sample_error(actual_label: np.ndarray, log_label_probability: np.ndarray,
                                         inv_log_label_probability: np.ndarray) -> float:
        """
        Counts error between earlier prepared logarithmized label probability and actual label.

        :param actual_label: true data label
        :param log_label_probability: label probability with `log(x)` function used on it
        :param inv_log_label_probability: label probability with `(1 - log(x))` function used on it
        :return: error of single data sample
        """
        first_component = actual_label @ np.transpose(log_label_probability)
        second_component = (1 - actual_label) @ np.transpose(inv_log_label_probability)
        error = -(first_component + second_component)
        return error
