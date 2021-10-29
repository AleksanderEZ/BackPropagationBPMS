from Layer import Layer
from InputLayer import InputLayer
from OutputLayer import OutputLayer
from NeuralNetwork import NeuralNetwork
import tensorflow as tf

test_input = tf.constant([1, -2], dtype='float32')
training_X = tf.constant([[2, 2], [2, 5], [4, 3], [4, 4], [0, 3], [3, 0], [4, 6], [6, 2]], dtype='float32')
training_y = tf.constant([1, 1, 1, 1, 0, 0, 0, 0], dtype='float32')

network = NeuralNetwork(0.01, test_input.shape[0])
network.add_layer(InputLayer(4))
network.add_layer(Layer(3))
network.add_layer(OutputLayer(1))

print(network.predict(test_input))
print(network.layers)

network.fit(training_X, training_y, 100)
network.predict(training_X[2])
