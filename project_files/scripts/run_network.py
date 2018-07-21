import mnist
import numpy as np

from project_files.neural_network.activation_functions import SigmoidFunction
from project_files.neural_network.network_layers import FullyConnectedLayer, FlatteningLayer
from project_files.neural_network.neural_network import NeuralNetworkBuilder

np.seterr(all='raise')
train_data_x = mnist.train_images()
train_data_y = mnist.train_labels()

test_data_x = mnist.test_images()
test_data_y = mnist.test_labels()

# train_data_x = np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
#                          [0.5, 0.7, 0.8, 0.2, 0.0, 0.0],
#                          [0.3, 0.1, 0.4, 0.0, 0.0, 0.0],
#                          [0.0, 0.1, 0.2, 0.5, 0.6, 0.5],
#                          [0.0, 0.0, 0.0, 0.3, 0.7, 0.4],
#                          [0.0, 0.2, 0.1, 0.8, 0.9, 0.0]])
# train_data_y = np.array([1, 1, 1, 0, 0, 0])
#
# test_data_x = np.array([[0.5, 0.2, 0.3, 0.1, 0.1, 0.0],
#                         [0.9, 0.9, 0.0, 0.0, 0.3, 0.2],
#                         [1, 1, 1, 0, 0, 0],
#                         [0, 0, 0, 1, 1, 1],
#                         [0.0, 0.0, 0.0, 0.9, 0.9, 0.9]])
# test_data_y = np.array([1, 1, 1, 0, 0])

shape = train_data_x[0].shape

neural_network = (NeuralNetworkBuilder()
                  .add_layer(FlatteningLayer())
                  .add_layer(FullyConnectedLayer(50, SigmoidFunction))
                  .add_layer(FullyConnectedLayer(50, SigmoidFunction))
                  .add_layer(FullyConnectedLayer(10, SigmoidFunction, True))
                  .build(shape))

neural_network.teach_network(train_data_x, train_data_y, 50, learning_rate=1)
after = neural_network.predict(train_data_x)

print(after)
print(train_data_y)
print(np.sum(after == train_data_y) / len(after))

after = neural_network.predict(test_data_x)
print(after)
print(test_data_y)
print(np.sum(after == test_data_y) / len(after))
