import numpy

from project_files.image_parser_files.create_train_data import create_train_data
from project_files.network_files.neural_network import NeuralNetworkDirector, NeuralNetworkBuilder

train_data_x, train_data_y = create_train_data("images\\small_set", 50)

network_builder = NeuralNetworkBuilder()
network_director = NeuralNetworkDirector(network_builder)
neural_network = network_director.construct()

before = neural_network.predict(train_data_x)
neural_network.teach_network(train_data_x, train_data_y)
after = neural_network.predict(train_data_x)

print(before)
print(after)
# print(train_data_y)
# n = numpy.array([[[3, 4, 8], [3, 55, 8]], [[99, 3, 2], [322, 2, 22]]])
# print(n)
