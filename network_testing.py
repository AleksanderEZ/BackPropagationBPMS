from Layer import Layer
from InputLayer import InputLayer
from OutputLayer import OutputLayer
from NeuralNetwork import NeuralNetwork
import numpy as np

# Data
training_X = np.array([[2, 2], [2, 5], [4, 3], [4, 4], [0, 3], [3, 0], [4, 6], [6, 2]], dtype='float32')
training_y = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype='float32')

# Model
network = NeuralNetwork(0.1, training_X.shape[1])
network.add_layer(InputLayer(4))
network.add_layer(Layer(3))
network.add_layer(OutputLayer(1))

# Training
network.fit(training_X, training_y, 10000, validation_X=training_X, validation_Y=training_y, batch_size=training_X.shape[0])

# Validation
random_permutation = np.random.permutation(training_X.shape[0])
shuffledX = training_X[random_permutation]
shuffledY = training_y[random_permutation]
for i in range(training_X.shape[0]):
    result = np.round(network.predict(shuffledX[i]))
    print("Resultado obtenido:", result, "vs esperado:", shuffledY[i])
result = np.round(network.predict(np.array([3,3])))
print("Resultado obtenido:", result, "vs esperado: 1")
