from Layer import Layer
import tensorflow as tf


class OutputLayer(Layer):
    def adjust_weights_and_biases(self, y, eta, prev_layer):
        print(prev_layer.activated_neurons)
        self.error = (self.activated_neurons - y)*(self.activated_neurons*(1-self.activated_neurons))
        self.biases = self.biases -eta * self.error
        self.weights = self.weights -eta * prev_layer.activated_neurons * self.error

        print("Error =", self.error.numpy())