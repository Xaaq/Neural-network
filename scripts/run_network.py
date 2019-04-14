"""
Script that shows sample functionalities of `NeuralNetwork` class.
"""
from typing import Tuple

import mnist
import numpy as np

from src.network_functions.activation_functions import SigmoidFunction
from src.layer_tools.layer_implementations import FullyConnectedLayer, FlatteningLayer
from src.neural_network.neural_network import NeuralNetworkBuilder, NeuralNetwork
from src.data_processing.data_processor import DataProcessor
from src.data_processing.dataset import Dataset


def prepare_datasets() -> Tuple[Dataset, Dataset]:
    """
    Gets MNIST train and test image datasets, wraps them into :class:`Dataset` classes and returns them.

    :return: tuple of train and test datasets
    """
    train_data_x = mnist.train_images()
    train_data_y = mnist.train_labels()

    test_data_x = mnist.test_images()
    test_data_y = mnist.test_labels()

    data_processor = DataProcessor()
    train_dataset = Dataset(data_processor.normalize_data(train_data_x),
                            data_processor.convert_label_vector_to_matrix(train_data_y))
    test_dataset = Dataset(data_processor.normalize_data(test_data_x),
                           data_processor.convert_label_vector_to_matrix(test_data_y))

    return test_dataset, train_dataset


def build_network(shape: Tuple[int, ...]) -> NeuralNetwork:
    """
    Creates :class:`NeuralNetwork` instance.

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


def predict_and_print_results(neural_network: NeuralNetwork, type_of_data: str, dataset: Dataset):
    """
    Predicts given data using provided neural network and prints results.

    :param neural_network: network to use
    :param type_of_data: type of used data ("test" or "train")
    :param dataset: dataset on which to make predictions
    """
    predicted_dataset = neural_network.predict(dataset)

    number_of_correct_labels = np.sum(dataset.label_vector == predicted_dataset.label_vector)
    prediction_accuracy = number_of_correct_labels * 100 / len(predicted_dataset.label_vector)

    print(f"Accuracy on {type_of_data} data: {prediction_accuracy:.2f}%")


def main():
    """
    Shows functionality of :class:`NeuralNetwork` class.
    """
    test_dataset, train_dataset = prepare_datasets()
    single_data_sample_shape = train_dataset.data[0].shape

    neural_network = build_network(single_data_sample_shape)
    neural_network.fit(train_dataset, 100, learning_rate=1)

    predict_and_print_results(neural_network, "train", train_dataset)
    predict_and_print_results(neural_network, "test", test_dataset)


if __name__ == "__main__":
    main()
