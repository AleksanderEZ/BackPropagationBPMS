from Layer import Layer
import tensorflow as tf


class OutputLayer(Layer):
    def adjust_weights_and_biases(self, y, eta, prev_layer):
        print(prev_layer.activated_neurons)
        print("Output weights", self.weights)
        print("Output biases", self.biases)
        print("Previous layer activated neurons", prev_layer.activated_neurons)
        self.error = (self.activated_neurons - y)*(self.activated_neurons*(1-self.activated_neurons))
        self.biases = self.biases -eta * self.error
        self.weights = self.weights -eta * tf.reshape(prev_layer.activated_neurons, (self.weights.shape[0], 1)) * self.error
        print("Adjusted output weights", self.weights)
        print("Adjusted output biases", self.biases)

        print("Error =", self.error)