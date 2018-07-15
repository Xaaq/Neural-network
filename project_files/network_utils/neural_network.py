"""
Module containing neural network class and things needed to create it - builder and director.
"""

import numpy

from project_files.network_utils.network_layers import AbstractLayer


# TODO: zastosowac podobna notacje co nizej co do docstringow rzeczy ktore maja wzory
# .. math ::
#         E = \\sum_{j=0}^k |p(x_j) - y_j|^2
# TODO: zrobic testy
# TODO: jesli sie da to z klas wywalic inity i przerzucic zmienne inicjalizowane w nich do ciala klasy (OSTROŻNIE! to zmienne statyczne wtedy beda)
# TODO: przeskanowac wszystko pylintem
# TODO: zrobic type hinty i wyrzucic je z docstringow
# TODO: sprawdzic czy nie ma lepszego sposobu na budowanie sieci niz uzywanie metod z NeuralNetwork
# TODO: sprawdzic w calym projekcie komentarze (szczegolnie pod katem tego czy jest w nich slowo "image", ew. zastapic "data sample"
# TODO: zamienic fory na list comprehension
# TODO: dodac setery i getery za pomoca @property
# TODO: zmienic ustawienia pycharma zeby formatowalo pod pep8
# TODO: zamienic mnozenie macierzy na symbol "@" i zobaczyc czy jest on szybszy od mnozenia za pomoca metody "dot" (chyba tak sie nazywala)
# TODO: zobaczyc czy da sie cos zrobic z jupyter notebook (w sensie czy pasowalby on tu do projektu)
# TODO: dodac requirements.txt
# TODO: zmienic wszystkie numpy. na np.
class NeuralNetwork:
    """
    Class used to do operations on neural network. It can do actions on it like learning and predicting learned classes.
    To create instance of this class use :class:`NeuralNetworkDirector`.
    """

    def __init__(self):
        """
        Initializes empty layer list for this neural network.
        """
        self.__layer_list = []

    def add_layer(self, layer_to_add):
        """
        Adds layer to this network. This method returns this object so can be chained with itself.

        :param layer_to_add: layer to add to network
        :return: self
        """
        self.__layer_list.append(layer_to_add)
        return self

    def initialize_layers(self, input_data_dimensions):
        """
        Initializes all layers in this network. This method should be called after all needed layers have been added to
        the network.

        :param input_data_dimensions: tuple of dimensions of single input data sample
        :type input_data_dimensions: tuple of int
        """
        next_layer_dimensions = input_data_dimensions

        for layer in self.__layer_list:
            next_layer_dimensions = layer.initialize_layer(next_layer_dimensions)

    # TODO: zobaczyc czy alphe dac jako arguent tej metody czy jako jakas zmienna tej klasy
    def teach_network(self, input_data, data_labels):
        """
        Teaches neural network on given data.

        :param input_data: data on which network has to learn on, format of data is multi-dimensional matrix:\n
            `number of input images x number of channels in image x width of single image x height of single image`
        :param data_labels: labels of input data, format of this is vector of labels:\n
            `number of input images x 1`
        """
        # TODO: dodac do docstring returna (albo i nie) i dodac parametr z iloscia iteracji uczenia
        normalized_data = self.__normalize_data(input_data)

        for _ in range(500):
            data_after_forward_pass = self.__forward_propagation(normalized_data)
            subtracted_data = data_after_forward_pass - data_labels
            self.__backward_propagation(subtracted_data)
            self.__update_weights()

            cost = self.__count_cost(data_after_forward_pass, data_labels)
            # TODO: zrobic cos z tym printem (albo log albo nie wiem)
            print(cost)

    def predict(self, input_data):
        """
        Predicts output classes of input data.

        :param input_data: data to predict
        :return: output classes for every data sample
        """
        normalized_data = self.__normalize_data(input_data)
        output_data = self.__forward_propagation(normalized_data)
        rounded_output_data = numpy.round(output_data)
        return rounded_output_data

    def compute_numerical_gradient(self, input_data, data_labels):
        """
        Computes gradient of weights in this network by counting it numerical way. This method is very slow and should
        be used only to check if gradient counted by other methods is computed correctly.

        :param input_data: data on which compute gradient
        :param data_labels: labels of input data
        :return: gradient of weights in this network
        """
        # TODO: dokonczyc

    @staticmethod
    def __normalize_data(data_to_normalize):
        """
        Normalizes given matrix - transforms values in it to range [0, 1].

        :param data_to_normalize: data to process
        :return: normalized data
        """
        # TODO: zrobic cos z tym bo to nie normalizuje w taki sposob jak napotkalo dane uczace, tylko zawsze na podstawie aktualnych danych
        max_number = numpy.max(data_to_normalize)
        min_number = numpy.min(data_to_normalize)
        difference = max_number - min_number
        normalized_data = (data_to_normalize - min_number) / difference
        return normalized_data

    def __forward_propagation(self, input_data):
        """
        Does forward propagation for every layer in this network based on given data.

        :param input_data: data on which to make forward pass
        :return: output data of network
        """
        data_for_next_layer = input_data

        for layer in self.__layer_list:
            data_for_next_layer = layer.forward_propagation(data_for_next_layer)

        return data_for_next_layer

    def __backward_propagation(self, input_data):
        """
        Does backward propagation for every layer in this network based on given data.

        :param input_data: data that are output of neural network used to make backward pass
        """
        data_for_previous_layer = input_data

        for layer in reversed(self.__layer_list):
            data_for_previous_layer = layer.backward_propagation(data_for_previous_layer)

    def __update_weights(self):
        """
        Updates weights in all layers in this network based on data from forward and backward propagation.
        """
        for layer in self.__layer_list:
            layer.update_weights()

    @staticmethod
    def __count_cost(network_output_data, data_labels):
        """
        Counts cost of learned data.

        :param network_output_data: predicted data outputted by neural network
        :param data_labels: labels of data
        :return: cost of learned data
        """
        data_count, _ = numpy.shape(network_output_data)

        first_component = numpy.dot(numpy.transpose(data_labels),
                                    numpy.log(network_output_data))
        second_component = numpy.dot(1 - numpy.transpose(data_labels),
                                     numpy.log(1 - network_output_data))
        cost = -(first_component + second_component) / data_count
        # TODO: zobaczyc czy da sie cos zrobic z rym [0][0]
        return cost[0]


class NeuralNetworkBuilder:
    """
    Builder used to build neural network with given layers.
    """

    def __init__(self):
        """
        Initializes empty neural network.
        """
        self.__neural_network = NeuralNetwork()

    def add_layer(self, layer_to_add: AbstractLayer) -> "NeuralNetworkBuilder":
        """
        Adds layer to neural network that's being built.

        :param layer_to_add: layer to add to network
        :return: this builder instance, so this method can be chained
        """
        self.__neural_network.add_layer(layer_to_add)
        return self

    def build(self) -> NeuralNetwork:
        """
        Initializes and returns neural network with earlier provided layers.

        :return: built neural network
        """
        input_data_dimensions = 6
        self.__neural_network.initialize_layers(input_data_dimensions)
        return self.__neural_network
