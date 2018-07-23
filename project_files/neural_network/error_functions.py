"""
Module containing error functions used in neural networks.
"""
from abc import abstractmethod

import numpy as np


class AbstractErrorFunction:
    """
    Base class for types of error functions.
    """

    @staticmethod
    @abstractmethod
    def count_error(predicted_label_probabilities: np.ndarray, actual_labels: np.ndarray) -> float:
        """
        Counts error of provided data.

        :param predicted_label_probabilities: predicted data labels probabilities
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

    @staticmethod
    def count_error(predicted_label_probabilities: np.ndarray, actual_labels: np.ndarray) -> float:
        data_samples_count = np.shape(predicted_label_probabilities)[0]
        logarithmic_network_output_data = np.transpose(np.log(predicted_label_probabilities))
        inverse_logarithmic_network_output_data = np.transpose(np.log(1 - predicted_label_probabilities))
        error_sum = 0
        # TODO: rozbic to na metody
        for index in range(data_samples_count):
            data_label_sample = actual_labels[index, :]
            logarithmic_network_output_data_sample = logarithmic_network_output_data[:, index]
            inverse_logarithmic_network_output_data_sample = inverse_logarithmic_network_output_data[:, index]

            first_component = data_label_sample @ logarithmic_network_output_data_sample
            second_component = (1 - data_label_sample) @ inverse_logarithmic_network_output_data_sample
            error = -(first_component + second_component)
            error_sum += error

        error_sum /= data_samples_count
        return error_sum
