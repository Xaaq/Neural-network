"""
Module containing container class for datasets.
"""
from typing import Optional

import numpy as np


class Dataset:
    """
    Data class containing data and their labels in form of vectors for every data sample.
    """

    def __init__(self, data: np.ndarray, label_matrix: Optional[np.ndarray] = None):
        self.__data: np.ndarray = data
        self.__label_matrix: Optional[np.ndarray] = label_matrix

    @property
    def data(self) -> np.ndarray:
        """
        Returns data samples.

        :return: data samples
        """
        return self.__data

    @property
    def label_matrix(self) -> Optional[np.ndarray]:
        """
        Returns labels in form of matrix.

        :return: label matrix
        """
        return self.__label_matrix

    @property
    def label_vector(self) -> Optional[np.ndarray]:
        """
        Converts matrix of labels to label vector and returns it. For example converts following matrix:
        ::
            [[0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 1, 0, 0, 0],
             [1, 0, 0, 0, 0]]

        To following vector:
        ::
            [1, 3, 2, 1, 0]

        :return: vector of labels
        """
        output_class_vector = np.argmax(self.__label_matrix, 1) if self.__label_matrix is not None else None
        return output_class_vector