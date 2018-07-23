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


class CrossEntropyCostFunction(AbstractCostFunction):
    """
    Class that's implementing following cost function:
        :math:`cost = -(y log(p) + (1 - y) log(1 - p))`
    where:
        * p - predicted probability of label
        * y - true value of label
    """

    @staticmethod
    def count_cost(predicted_label_probabilities: np.ndarray, actual_labels: np.ndarray) -> float:
        data_samples_count = np.shape(predicted_label_probabilities)[0]
        logarithmic_network_output_data = np.transpose(np.log(predicted_label_probabilities))
        inverse_logarithmic_network_output_data = np.transpose(np.log(1 - predicted_label_probabilities))
        cost_sum = 0

        for index in range(data_samples_count):
            data_label_sample = actual_labels[index, :]
            logarithmic_network_output_data_sample = logarithmic_network_output_data[:, index]
            inverse_logarithmic_network_output_data_sample = inverse_logarithmic_network_output_data[:, index]

            first_component = data_label_sample @ logarithmic_network_output_data_sample
            second_component = (1 - data_label_sample) @ inverse_logarithmic_network_output_data_sample
            cost = -(first_component + second_component)
            cost_sum += cost

        cost_sum /= data_samples_count
        return cost_sum
