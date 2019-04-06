import mnist
import numpy as np

from project_files.neural_network.activation_functions import SigmoidFunction
from project_files.neural_network.network_layers import FullyConnectedLayer, FlatteningLayer
from project_files.neural_network.neural_network import NeuralNetworkBuilder

train_data_x = mnist.train_images()
train_data_y = mnist.train_labels()

test_data_x = mnist.test_images()
test_data_y = mnist.test_labels()

shape = train_data_x[0].shape

neural_network = (NeuralNetworkBuilder()
                  .set_layers([FlatteningLayer(),
                               FullyConnectedLayer(50, SigmoidFunction()),
                               FullyConnectedLayer(50, SigmoidFunction()),
                               FullyConnectedLayer(10, SigmoidFunction())])
                  .build(shape))
neural_network.fit(train_data_x, train_data_y, 100, learning_rate=1)
after = neural_network.predict(train_data_x)

print(after)
print(train_data_y)
print(np.sum(after == train_data_y) / len(after))

after = neural_network.predict(test_data_x)
print(after)
print(test_data_y)
print(np.sum(after == test_data_y) / len(after))
