from Layer import Layer
from InputLayer import InputLayer
from OutputLayer import OutputLayer
from NeuralNetwork import NeuralNetwork
import numpy as np

test_input = np.array([1, -2], dtype='float32')
training_X = np.array([[2, 2], [2, 5], [4, 3], [4, 4], [0, 3], [3, 0], [4, 6], [6, 2]], dtype='float32')
training_y = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype='float32')

network = NeuralNetwork(0.1, test_input.shape[0])
network.add_layer(InputLayer(4))
network.add_layer(Layer(3))
network.add_layer(OutputLayer(1))

print(network.predict(test_input))
print(network.layers)

network.fit(training_X, training_y, 10000)
for datapoint in training_X:
    result = np.round(network.predict(datapoint))
    print(result)
