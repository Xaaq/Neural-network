"""
Script that shows sample functionalities of `NeuralNetwork` class.
"""
from typing import Tuple

import mnist
import numpy as np

from src.neural_network.activation_functions import SigmoidFunction
from src.neural_network.network_layers import FullyConnectedLayer, FlatteningLayer
from src.neural_network.neural_network import NeuralNetworkBuilder, NeuralNetwork
from src.utils.data_processor import DataProcessor


def build_network(shape: Tuple[int, ...]) -> NeuralNetwork:
    """
    Creates `NeuralNetwork` instance.

    :param shape: shape of single data sample
    :return: built `NeuralNetwork` instance
    """
    neural_network = (NeuralNetworkBuilder()
                      .set_layers([FlatteningLayer(),
                                   FullyConnectedLayer(50, SigmoidFunction()),
                                   FullyConnectedLayer(50, SigmoidFunction()),
                                   FullyConnectedLayer(10, SigmoidFunction())])
                      .build(shape))
    return neural_network


def predict_and_print_results(neural_network: NeuralNetwork, type_of_data: str, data_samples: np.ndarray,
                              label_vector: np.ndarray):
    """
    Predicts given data using provided neural network and prints results.

    :param neural_network: network to use
    :param type_of_data: type of used data ("test" or "train")
    :param data_samples: data samples
    :param label_vector: vector of data labels
    """
    # TODO: fix this function
    data_processor = DataProcessor()
    predicted_data = neural_network.predict(data_samples)
    processed_predicted_data = data_processor.convert_label_matrix_to_vector(predicted_data)
    number_of_correct_labels = np.sum(processed_predicted_data == label_vector)
    prediction_accuracy = number_of_correct_labels * 100 / len(processed_predicted_data)
    print(f"Accuracy on {type_of_data} data: {prediction_accuracy:.2f}%")


def main():
    """
    Shows functionality of `NeuralNetwork` class.
    """
    train_data_x = mnist.train_images()
    train_data_y = mnist.train_labels()

    test_data_x = mnist.test_images()
    test_data_y = mnist.test_labels()

    data_processor = DataProcessor()
    number_of_labels = 10
    preprocessed_train_data_x, preprocessed_train_data_y = data_processor.preprocess_data(train_data_x, train_data_y,
                                                                                          number_of_labels)
    preprocessed_test_data_x, preprocessed_test_data_y = data_processor.preprocess_data(test_data_x, test_data_y,
                                                                                        number_of_labels)

    shape = preprocessed_train_data_x[0].shape

    neural_network = build_network(shape)
    neural_network.fit(preprocessed_train_data_x, preprocessed_train_data_y, 100, learning_rate=1)

    predict_and_print_results(neural_network, "train", preprocessed_train_data_x, train_data_y)
    predict_and_print_results(neural_network, "test", preprocessed_test_data_x, test_data_y)


if __name__ == "__main__":
    main()
