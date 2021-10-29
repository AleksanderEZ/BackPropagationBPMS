from Layer import Layer
import tensorflow as tf


class InputLayer(Layer):
    def adjust_weights_and_biases(self, eta, input, next_layer):
        self.error = (tf.tensordot(tf.transpose(next_layer.weights), next_layer.error, 1))*(self.activated_neurons*(1-self.activated_neurons))
        self.biases = self.biases -eta * self.error
        self.weights = self.weights -eta * input * self.error