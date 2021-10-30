import numpy as np


class Layer:
    def __init__(self, num_neurons):
        self.neurons = np.array([0 for i in range(num_neurons)], dtype='float32')
        self.activated_neurons = None
        self.weights = None
        self.biases = np.array(np.random.rand(num_neurons), dtype='float32')
        self.error = None

    def initialize_weights(self, prev_layer_num_neurons):
        self.weights = np.array(np.random.rand(prev_layer_num_neurons, self.neurons.shape[0]), dtype='float32')

    def compute_values(self, prev_neurons):
        # Multiplicamos los pesos por los valores de las neuronas de la capa anterior y los guardamos en las neuronas de esta capa
        self.neurons = np.matmul(prev_neurons, self.weights) + self.biases
        # Pasamos las neuronas por la función de activación de la capa
        self.activated_neurons = self.activate()

    def activate(self):
        return 1/(1 + np.exp(-self.neurons))

    def adjust_weights_and_biases(self, eta, prev_layer):
        self.biases = self.biases - eta * self.error
        self.weights = self.weights - eta * np.tensordot(prev_layer.activated_neurons, self.error, 0)

    def compute_error(self, next_layer):
        if next_layer.error.shape == (1,):
            self.error = (np.transpose(next_layer.weights) * next_layer.error) * (
                        self.activated_neurons * (1 - self.activated_neurons))
        else:
            self.error = (np.transpose(np.matmul(next_layer.weights, np.transpose(next_layer.error))))*(self.activated_neurons*(1-self.activated_neurons))
        self.error = self.error.reshape(self.biases.shape)
