import tensorflow as tf


class Layer:
    def __init__(self, num_neurons):
        self.neurons = tf.Variable(tf.constant([0 for i in range(num_neurons)], dtype='float32'))
        self.activated_neurons = None
        self.weights = None
        self.biases = tf.Variable(tf.random.uniform([num_neurons], dtype='float32'))
        self.error = None

    def initialize_weights(self, prev_layer_num_neurons):
        self.weights = tf.Variable(tf.random.uniform([prev_layer_num_neurons, self.neurons.shape[0]], dtype='float32'))

    def compute_values(self, prev_neurons):
        # Multiplicamos los pesos por los valores de las neuronas de la capa anterior y los guardamos en las neuronas de esta capa
        self.neurons = tf.tensordot(prev_neurons, self.weights, 1) + self.biases
        # Pasamos las neuronas por la función de activación de la capa
        self.activated_neurons = self.activate()

    def activate(self):
        return tf.math.sigmoid(self.neurons)

    def adjust_weights_and_biases(self, y, eta, prev_layer, next_layer):
        self.error = (tf.tensordot(tf.transpose(next_layer.weights), next_layer.error, 1))*(self.activated_neurons*(1-self.activated_neurons))
        self.biases = self.biases -eta * self.error
        self.weights = self.weights -eta * prev_layer.activated_neurons * self.error