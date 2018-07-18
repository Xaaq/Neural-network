import sys

from tqdm import tqdm


class NeuralNetworkProgressBar(tqdm):
    __column_width = 80
    __bar_format = "Learning progress: [{bar}]     Remaining time: {remaining}s     Cost: {desc}"

    def __init__(self, iteration_count):
        super().__init__(range(iteration_count),
                         file=sys.stdout,
                         ncols=self.__column_width,
                         bar_format=self.__bar_format)

    def update_cost(self, cost: float):
        formatted_cost = "{0:.4f}".format(cost)
        self.set_description_str(formatted_cost)
        self.ncols = self.__column_width + len(formatted_cost)
