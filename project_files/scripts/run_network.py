import numpy as np

from project_files.network_utils.network_layers import FullyConnectedLayer
from project_files.network_utils.neural_network import NeuralNetworkBuilder

# train_data_x, train_data_y = create_train_data("images\\small_set", 50)

neural_network = (NeuralNetworkBuilder()
                  .add_layer(FullyConnectedLayer(10))
                  .add_layer(FullyConnectedLayer(1))
                  .build((6,)))

train_data_x = np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                         [0.5, 0.7, 0.8, 0.2, 0.0, 0.0],
                         [0.3, 0.1, 0.4, 0.0, 0.0, 0.0],
                         [0.0, 0.1, 0.2, 0.5, 0.6, 0.5],
                         [0.0, 0.0, 0.0, 0.3, 0.7, 0.4],
                         [0.0, 0.2, 0.1, 0.8, 0.9, 0.0]])
train_data_y = np.array([[1], [1], [1], [0], [0], [0]])

test_data = np.array([[0.5, 0.2, 0.3, 0.1, 0.1, 0.0],
                      [0.9, 0.9, 0.0, 0.0, 0.3, 0.2],
                      [1, 1, 1, 0, 0, 0],
                      [0, 0, 0, 1, 1, 1],
                      [0.0, 0.0, 0.0, 0.9, 0.9, 0.9]])
before = neural_network.predict(test_data)
neural_network.teach_network(train_data_x, train_data_y)
after = neural_network.predict(test_data)

print(before == np.array([[1], [1], [1], [0], [0]]))
print(after == np.array([[1], [1], [1], [0], [0]]))
# print(train_data_y)
# n = numpy.array([[[3, 4, 8], [3, 55, 8]], [[99, 3, 2], [322, 2, 22]]])
# print(n)
