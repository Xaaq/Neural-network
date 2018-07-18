import numpy as np

from project_files.neural_network.activation_functions import SigmoidFunction
from project_files.neural_network.network_layers import FullyConnectedLayer
from project_files.neural_network.neural_network import NeuralNetworkBuilder

neural_network = (NeuralNetworkBuilder()
                  .add_layer(FullyConnectedLayer(10, SigmoidFunction))
                  .add_layer(FullyConnectedLayer(2, SigmoidFunction))
                  .build((6,)))

train_data_x = np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                         [0.5, 0.7, 0.8, 0.2, 0.0, 0.0],
                         [0.3, 0.1, 0.4, 0.0, 0.0, 0.0],
                         [0.0, 0.1, 0.2, 0.5, 0.6, 0.5],
                         [0.0, 0.0, 0.0, 0.3, 0.7, 0.4],
                         [0.0, 0.2, 0.1, 0.8, 0.9, 0.0]])
train_data_y = np.array([1, 1, 1, 0, 0, 0])

test_data_x = np.array([[0.5, 0.2, 0.3, 0.1, 0.1, 0.0],
                        [0.9, 0.9, 0.0, 0.0, 0.3, 0.2],
                        [1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1],
                        [0.0, 0.0, 0.0, 0.9, 0.9, 0.9]])

before = neural_network.predict(test_data_x)
neural_network.teach_network(train_data_x, train_data_y, 150)
after = neural_network.predict(test_data_x)

print(before)
print(after)
