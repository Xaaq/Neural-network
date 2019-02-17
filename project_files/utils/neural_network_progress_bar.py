"""
This module contains progress bar used in neural network.
"""
import sys

from tqdm import tqdm


class NeuralNetworkProgressBar(tqdm):
    """
    This is wrapper class for tqdm progress bar that is used to indicate progress of neural network learning.
    """
    __column_width = 90
    __bar_format = "Learning progress: [{bar}]     Remaining time: {remaining}s     Learning error: {desc}"

    def __init__(self, iteration_count: int):
        """
        Initializes this progress bar.

        :param iteration_count: number of iterations on which this progress bar will operate
        """
        super().__init__(range(iteration_count),
                         file=sys.stdout,
                         ncols=self.__column_width,
                         bar_format=self.__bar_format)

    def update_error(self, error: float):
        """
        Updates error function value in this progress bar description.

        :param error: error function value
        """
        formatted_error = f"{error:.4f}"
        self.set_description_str(formatted_error)
        self.ncols = self.__column_width + len(formatted_error)
