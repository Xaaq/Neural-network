"""
Module containing class that provides tools to pre and post-process data fed to neural network.
"""
from typing import Optional

import numpy as np


class DataProcessor:
    """
    Class that provides tools to pre-process data that are input to neural network and post-process its output.
    """

    def __init__(self):
        self.__data_mean: Optional[np.ndarray] = None
        self.__data_std_dev: Optional[np.ndarray] = None
        self.__number_of_labels: Optional[int] = None

    def normalize_data(self, data_to_normalize: np.ndarray) -> np.ndarray:
        """
        Normalizes given matrix - transforms values in it to range around [-1, 1]. When first used, it remembers data
        normalization coefficients, so firstly use it on train set and then on test set.

        :param data_to_normalize: data to process
        :return: normalized data
        """
        if self.__data_mean is None or self.__data_std_dev is None:
            self.__data_mean = np.mean(data_to_normalize, 0)
            self.__data_std_dev = np.std(data_to_normalize, 0)

        data_with_subtracted_mean = data_to_normalize - self.__data_mean
        normalized_data = np.divide(data_with_subtracted_mean, self.__data_std_dev, where=self.__data_std_dev != 0)
        return normalized_data

    def convert_label_vector_to_matrix(self, label_vector: np.ndarray) -> np.ndarray:
        """
        Converts vector of labels to matrix representation. First time it is used, it deduces number of labels based on
        highest data label and saves it for future calls. Deducing number of labels is based on value of highest label
        value e.g. if highest label has value of 4, then output matrix will have 5 labels (because counting from 0).

        For example when 5 labels will be deduced, it converts following vector:
        ::
            [1, 3, 4, 1, 0]

        To following matrix:
        ::
            [[0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 0, 0, 1],
             [0, 1, 0, 0, 0],
             [1, 0, 0, 0, 0]]

        :param label_vector: vector of labels
        :return: matrix of labels
        """
        if self.__number_of_labels is None:
            self.__number_of_labels = label_vector.max() + 1

        label_matrix = []

        for label_value in label_vector:
            row = np.zeros(self.__number_of_labels)
            row[label_value] = 1
            label_matrix.append(row)

        return np.array(label_matrix)
