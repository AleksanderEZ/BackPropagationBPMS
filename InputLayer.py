from Layer import Layer
import numpy as np


class InputLayer(Layer):
    def adjust_weights_and_biases(self, eta, input):
        self.biases = self.biases - eta * self.error
        self.weights = self.weights - eta * np.tensordot(input, self.error, 0)