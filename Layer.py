import tensorflow as tf


class Layer:
    def __init__(self, num_neurons):
        self.neurons = tf.Variable(tf.constant([0 for i in range(num_neurons)], dtype='float32'))
        self.weights = None

    def initialize_weights(self, prev_layer_num_neurons):
        self.weights = tf.Variable(tf.random.uniform([prev_layer_num_neurons + 1, self.neurons.shape[0]], dtype='float32'))

    def compute_values(self, prev_neurons):
        values = tf.concat([tf.constant([1], dtype='float32'), prev_neurons], axis=0)   # Le añadimos el bias, que siempre es 1
        # Multiplicamos los pesos por los valores de las neuronas de la capa anterior y los guardamos en las neuronas de esta capa
        self.neurons = tf.tensordot(values, self.weights, 1)
        # Pasamos las neuronas por la función de activación de la capa
        self.neurons = self.activate()

    def activate(self):
        return tf.math.sigmoid(self.neurons)