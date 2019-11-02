# pylint: skip-file
import numpy as np

from src.network_core_tools.neural_layer_manager import NetworkLayerManager
from src.network_core_tools.neural_network import NeuralNetworkBuilder
from src.network_functions.activation_functions import SigmoidFunction
from src.network_functions.error_functions import CrossEntropyErrorFunction
from src.layer_tools.layer_implementations import FullyConnectedLayer, FlatteningLayer
from src.data_processing.data_processor import DataProcessor
from src.data_processing.dataset import Dataset
from src.utils.network_gradient_comparator import NetworkGradientComparator


def main():
    # train_data_x = mnist.train_images()[:100]
    # train_data_y = mnist.train_labels()[:100]
    #
    # test_data_x = mnist.test_images()
    # test_data_y = mnist.test_labels()

    train_data_x = np.array([[0.0, 0.0, 1.0, 1.0, 1.0],
                             [0.4, 0.3, 1.0, 0.0, 0.0],
                             [1.0, 0.5, 0.0, 0.0, 0.0],
                             [1.0, 1.0, 0.0, 0.0, 0.0]])

    train_data_y = np.array([1, 0, 0, 0])

    number_of_labels = 2
    shape = train_data_x[0].shape

    neural_network = (NeuralNetworkBuilder()
                      .set_layers([FlatteningLayer(),
                                   FullyConnectedLayer(30, SigmoidFunction()),
                                   FullyConnectedLayer(30, SigmoidFunction()),
                                   FullyConnectedLayer(30, SigmoidFunction()),
                                   FullyConnectedLayer(number_of_labels, SigmoidFunction())])
                      .build(shape))

    data_processor = DataProcessor()
    train_dataset = Dataset(data_processor.normalize_data(train_data_x),
                            data_processor.convert_label_vector_to_matrix(train_data_y))
    # test_dataset = Dataset(data_processor.normalize_data(test_data_x),
    #                        data_processor.convert_label_vector_to_matrix(test_data_y))
    # preprocessed_train_data_x, preprocessed_train_data_y = data_processor.preprocess_data(train_data_x, train_data_y,
    #                                                                                       number_of_labels)

    neural_network.fit(train_dataset, 100, learning_rate=1)
    after = neural_network.predict(train_dataset)

    number_of_correct_labels = np.sum(train_dataset.label_vector == after.label_vector)

    print(number_of_correct_labels / len(after.label_vector))

    # after = neural_network.predict(test_data_x)
    # print(after)
    # print(test_data_y)
    # print(np.sum(after == test_data_y) / len(after))

    layer_manager = NetworkLayerManager([FlatteningLayer(),
                                         FullyConnectedLayer(30, SigmoidFunction()),
                                         FullyConnectedLayer(30, SigmoidFunction()),
                                         FullyConnectedLayer(number_of_labels, SigmoidFunction())],
                                        shape)
    gradient_comparator = NetworkGradientComparator(layer_manager, CrossEntropyErrorFunction())
    magnitude_list = gradient_comparator.compute_gradient_difference_magnitudes(train_dataset)
    print(magnitude_list)


if __name__ == "__main__":
    main()
