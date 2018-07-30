"""
Module containing data containers used in neural network layers.
"""
import numpy as np

from project_files.utils.memento import WeightMemento


class GradientHelperData:
    """
    Data object used to store helper values that are byproduct of forward and backward propagation of neural network.
    They are used to count gradient of weights.
    """

    def __init__(self):
        """
        Initializes this data container data to empty values
        """
        self.__before_forward_multiplication: np.ndarray = None
        self.__before_backward_multiplication: np.ndarray = None

    def count_weight_gradient(self) -> np.ndarray:
        """
        Counts and returns gradient of weights based on data saved during forward and backward propagation.

        :return: gradient of weights of this layer
        """
        transposed_backward_data = np.transpose(self.before_backward_multiplication)
        weight_gradient = transposed_backward_data @ self.before_forward_multiplication

        number_of_data_samples = np.shape(self.before_forward_multiplication)[0]
        weight_gradient /= number_of_data_samples

        return weight_gradient

    @property
    def before_forward_multiplication(self) -> np.ndarray:
        """
        Getter for data before forward multiplication.

        :return: data before forward multiplication
        """
        return self.__before_forward_multiplication

    @before_forward_multiplication.setter
    def before_forward_multiplication(self, value: np.ndarray):
        """
        Setter for data before forward multiplication.

        :param value: new value for data before forward multiplication
        """
        self.__before_forward_multiplication = value

    @property
    def before_backward_multiplication(self) -> np.ndarray:
        """
        Getter for data before backward multiplication.

        :return: data before backward multiplication
        """
        return self.__before_backward_multiplication

    @before_backward_multiplication.setter
    def before_backward_multiplication(self, value: np.ndarray):
        """
        Setter for data before backward multiplication.

        :param value: new value for data before backward multiplication
        """
        self.__before_backward_multiplication = value


class WeightData:
    """
    Data object that encapsulates layer weights and allows to do computing on them.
    """

    def __init__(self, weight_dimensions: tuple):
        """
        Initializes this data container weights with provided dimensions.

        :param weight_dimensions: dimensions of weights that will be generated
        """
        self.__weights = self.__generate_random_weight_matrix(weight_dimensions)

    def update_weights(self, learning_rate: float, gradient_helper_data: GradientHelperData):
        """
        Updates weights using provided data.

        :param learning_rate: multiplier of weights update
        :param gradient_helper_data: additional data needed to compute weight gradient
        """
        self.__weights -= learning_rate * gradient_helper_data.count_weight_gradient()

    def save_weights(self) -> WeightMemento:
        """
        Saves weights in memento object.

        :return: memento object with saved weights
        """
        return WeightMemento(self.weights)

    def restore_weights(self, memento: WeightMemento):
        """
        Restores weights from memento object.

        :param memento: memento object from which to restore weight data
        """
        self.__weights = memento.get_weights()

    @property
    def weights(self) -> np.ndarray:
        """
        Returns stored weights.

        :return: stored weights
        """
        return self.__weights

    @staticmethod
    def __generate_random_weight_matrix(weight_dimensions: tuple) -> np.ndarray:
        """
        Randomly initializes weight matrix based on provided dimensions. All values in matrix are initialized in range
        [-0.5, 0.5].

        :param weight_dimensions: dimensions of weights to create
        :return: randomly initialized weight matrix
        """
        weights = np.random.rand(*weight_dimensions) - 0.5
        return weights
