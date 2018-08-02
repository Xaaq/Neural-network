"""
Module containing class that provides tools to pre and post-process data fed to neural network.
"""
from typing import Tuple

import numpy as np


class DataProcessor:
    """
    Class that provides tools to pre-process data that are input to neural network and post-process its output.
    """

    def preprocess_data(self, input_data: np.ndarray, label_vector: np.ndarray,
                        number_of_labels: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocesses input data by normalizing it and label vector by converting it to matrix.

        :param input_data: input data to normalize
        :param label_vector: vector of labels to change to matrix
        :param number_of_labels: number of possible labels
        :return: normalized input data, matrix of labels
        """
        normalized_data = self.normalize_data(input_data)
        label_matrix = self.convert_label_vector_to_matrix(label_vector, number_of_labels)
        return normalized_data, label_matrix

    @staticmethod
    def normalize_data(data_to_normalize: np.ndarray) -> np.ndarray:
        """
        Normalizes given matrix - transforms values in it to range [0, 1].

        :param data_to_normalize: data to process
        :return: normalized data
        """
        max_number = np.max(data_to_normalize)
        min_number = np.min(data_to_normalize)
        amplitude = max_number - min_number
        normalized_data = (data_to_normalize - min_number) / amplitude
        return normalized_data

    @staticmethod
    def convert_label_vector_to_matrix(label_vector: np.ndarray, number_of_labels: int) -> np.ndarray:
        """
        Converts vector of values (labels) to matrix representation. For example when 5 `number_of_labels` provided it
        converts following vector:

        :math:`[1, 3, 2, 1, 0]`

        To following matrix:
            [[0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 1, 0, 0, 0],
             [1, 0, 0, 0, 0]]

        :param label_vector: vector of labels
        :param number_of_labels: number of possible data labels
        :return: matrix of labels
        """
        # TODO: naprawic tego docstringa bo sie macierz zle wyswietla
        label_matrix = []

        for label_value in label_vector:
            row = np.zeros(number_of_labels)
            row[label_value] = 1
            label_matrix.append(row)

        return np.array(label_matrix)

    @staticmethod
    def convert_label_matrix_to_vector(label_matrix: np.ndarray) -> np.ndarray:
        """
        Converts matrix of values (labels) to vector representation. For example converts following matrix:

            [[0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 1, 0, 0, 0],
             [1, 0, 0, 0, 0]]

        To following vector:

        :math:`[1, 3, 2, 1, 0]`

        :param label_matrix: matrix of labels
        :return: vector of labels
        """
        # TODO: naprawic tego docstringa bo sie macierz zle wyswietla
        output_class_vector = np.argmax(label_matrix, axis=1)
        return output_class_vector
