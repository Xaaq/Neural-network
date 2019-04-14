"""
This module contains progress bar used in neural network.
"""
import sys
from typing import Iterable

from tqdm import tqdm


class NeuralNetworkProgressBar:
    """
    Wrapper class for tqdm progress bar, it provides simple interface used to indicate progress of neural network
    learning.
    """
    __column_width = 90

    def __init__(self, iteration_count: int):
        """
        Initializes this progress bar.

        :param iteration_count: number of iterations on which this progress bar will operate
        """
        bar_format = "Learning progress: [{bar}]     Remaining time: {remaining}s     Learning error: {desc}"
        self.__tqdm_bar: tqdm = tqdm(range(iteration_count), file=sys.stdout, ncols=self.__column_width,
                                     bar_format=bar_format)

    def update_error(self, error: float):
        """
        Updates error function value in this progress bar description.

        :param error: error function value
        """
        formatted_error = f"{error:.4f}"
        self.__tqdm_bar.set_description_str(formatted_error)
        self.__tqdm_bar.ncols = self.__column_width + len(formatted_error)

    def __iter__(self) -> Iterable[int]:
        return iter(self.__tqdm_bar)
