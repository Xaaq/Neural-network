"""
Script that shows sample functionalities of `NeuralNetwork` class.
"""
from typing import Tuple

import mnist
import numpy as np

from project_files.neural_network.activation_functions import SigmoidFunction
from project_files.neural_network.network_layers import FullyConnectedLayer, FlatteningLayer
from project_files.neural_network.neural_network import NeuralNetworkBuilder, NeuralNetwork


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
                              data_labels: np.ndarray):
    """
    Predicts given data using provided neural network and prints results.

    :param neural_network: network to use
    :param type_of_data: type of used data ("test" or "train")
    :param data_samples: data samples
    :param data_labels: data labels
    """
    predicted_data = neural_network.predict(data_samples)
    number_of_correct_labels = np.sum(predicted_data == data_labels)
    prediction_accuracy = number_of_correct_labels * 100 / len(predicted_data)
    print(f"Accuracy on {type_of_data} data: {prediction_accuracy}%")


def main():
    """
    Shows functionality of `NeuralNetwork` class.
    """
    train_data_x = mnist.train_images()[:1000]
    train_data_y = mnist.train_labels()[:1000]

    test_data_x = mnist.test_images()[:100]
    test_data_y = mnist.test_labels()[:100]

    shape = train_data_x[0].shape

    neural_network = build_network(shape)
    neural_network.fit(train_data_x, train_data_y, 100, learning_rate=1)

    predict_and_print_results(neural_network, "train", train_data_x, train_data_y)
    predict_and_print_results(neural_network, "test", test_data_x, test_data_y)


if __name__ == "__main__":
    main()
