from data_processing import load_data, preprocess
import numpy as np
from Layer import Layer
from InputLayer import InputLayer
from OutputLayer import OutputLayer
from NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt


data = load_data()

X = np.array(data.iloc[:, 1:13], dtype='float32')
y = np.array(data.iloc[:, 13], dtype='float32')

x_train, x_test, y_train, y_test = preprocess(X, y)

# Model
network = NeuralNetwork(0.1, input_size=12)
network.add_layer(InputLayer(12))
network.add_layer(Layer(8))
network.add_layer(OutputLayer(1))

# Training
history_error, history_accuracy = network.fit(x_train, y_train, epochs=250, validation_X=x_test, validation_Y=y_test, batches=64)

plt.subplot(1, 2, 1)
plt.plot(history_error)
plt.title('Error cuadrático medio')
plt.xlabel('Epoch')
plt.subplot(1, 2, 2)
plt.plot(history_accuracy)
plt.title('Precisión binaria')
plt.xlabel('Epoch')
plt.show()

for i in range(len(X)):
    print(np.round(network.predict(X[i])), y[i])