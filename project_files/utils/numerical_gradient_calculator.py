"""
Module containing numerical gradient calculator that allows to numerically compute gradient of neural network layers'
weights.
"""
import numpy as np

from project_files.neural_network.network_layers import FullyConnectedLayer
from project_files.neural_network.neural_network import NeuralNetwork


class NumericalGradientCalculator:
    def __init__(self, neural_network: NeuralNetwork):
        self.__neural_network = neural_network

    def compute_numerical_gradient(self, input_data: np.ndarray, data_labels: np.ndarray):
        """
        Computes gradient of weights in this network by counting it numerical way. This method is very slow and should
        be used only to check if gradient counted by other methods is computed correctly.

        :param input_data: data on which compute gradient
        :param data_labels: labels of input data
        :return: gradient of weights in this network
        """
        normalized_data = self.__neural_network._NeuralNetwork__data_processor.normalize_data(input_data)
        label_matrix = self.__neural_network._NeuralNetwork__data_processor.convert_label_vector_to_matrix(data_labels,
                                                                                                           self.__neural_network._NeuralNetwork__get_network_output_neuron_count())
        np.set_printoptions(linewidth=400)
        epsilon = 0.001
        nadmacierz = []
        nadmacierz2 = []
        for layer in self.__neural_network._NeuralNetwork__layer_list:
            if not isinstance(layer, FullyConnectedLayer):
                # TODO: obmyslic jak robic ten check (czy jakas inna klase abstrakcyjna, ktora mowi ze tej klasy thete mozna brac)
                continue
            shape = np.shape(layer.weight_data.weights)
            macierz = np.zeros(shape)
            weight_memento = layer.weight_data.save_weights()
            for weight_row in range(shape[0]):
                for weight_column in range(shape[1]):
                    layer.weight_data.weights[weight_row, weight_column] += epsilon

                    data_after_forward_pass = self.__neural_network._NeuralNetwork__forward_propagation(normalized_data)
                    error1 = self.__neural_network._NeuralNetwork__error_function.count_error(data_after_forward_pass,
                                                                                              label_matrix)
                    layer.weight_data.restore_weights(weight_memento)

                    layer.weight_data.weights[weight_row, weight_column] -= epsilon

                    data_after_forward_pass = self.__neural_network._NeuralNetwork__forward_propagation(normalized_data)
                    error2 = self.__neural_network._NeuralNetwork__error_function.count_error(data_after_forward_pass,
                                                                                              label_matrix)
                    layer.weight_data.restore_weights(weight_memento)

                    macierz[weight_row, weight_column] = (error1 - error2) / (2 * epsilon)
            nadmacierz.append(macierz)

            data_after_forward_pass = self.__neural_network._NeuralNetwork__forward_propagation(normalized_data)
            error_vector = data_after_forward_pass - label_matrix
            self.__neural_network._NeuralNetwork__backward_propagation(error_vector)
            nadmacierz2.append(layer._FullyConnectedLayer__gradient_calculator.count_weight_gradient())
            # a = layer._FullyConnectedLayer__count_weight_gradient()
            # print(a - macierz)
            # print(nadmacierz2[len(nadmacierz2) - 1] - nadmacierz[len(nadmacierz2) - 1])
            # print("==============================================================")

        # for layer in self.__layer_list:
        #     if not isinstance(layer, FullyConnectedLayer):
        #         continue
        #     layer._FullyConnectedLayer__weight_matrix = layer.weight_matrix_copy.copy()
        #     data_after_forward_pass = self.__forward_propagation(normalized_data)
        #     error_vector = data_after_forward_pass - label_matrix
        #     self.__backward_propagation(error_vector)
        #     nadmacierz2.append(layer._FullyConnectedLayer__count_weight_gradient())
        #  TODO: jesli ten for przetrwa to zmienic i na inna zmienna
        for i in range(len(nadmacierz)):
            print(nadmacierz2[i] - nadmacierz[i])
            print("==============================================================")
