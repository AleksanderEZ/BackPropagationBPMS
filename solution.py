from Layer import Layer
from NeuralNetwork import NeuralNetwork
import tensorflow as tf

test_input = tf.constant([1, -2], dtype='float32')
network = NeuralNetwork(0.01, test_input.shape[0])
network.add_layer(Layer(4))
network.add_layer(Layer(5))
network.add_layer(Layer(1))
print(network.predict(test_input))