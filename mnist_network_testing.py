from Layer import Layer
from InputLayer import InputLayer
from OutputLayer import OutputLayer
from NeuralNetwork import NeuralNetwork
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

# Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

img_size = x_train.shape[1]

x_train = x_train.reshape(x_train.shape[0], img_size*img_size)
x_test = x_test.reshape(x_test.shape[0], img_size*img_size)
input_shape = (img_size*img_size)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# Model
network = NeuralNetwork(0.1, input_shape)
network.add_layer(InputLayer(12))
network.add_layer(Layer(12))
network.add_layer(Layer(10))
network.add_layer(Layer(9))
network.add_layer(OutputLayer(10))

# Training
network.fit(x_train, y_train, 100, x_test, y_test, 128)

# # Validation
# random_permutation = np.random.permutation(training_X.shape[0])
# shuffledX = training_X[random_permutation]
# shuffledY = training_y[random_permutation]
# for i in range(training_X.shape[0]):
#     result = np.round(network.predict(shuffledX[i]))
#     print("Resultado obtenido:", result, "vs esperado:", shuffledY[i])
# result = np.round(network.predict(np.array([3,3])))
# print("Resultado obtenido:", result, "vs esperado: 1")
