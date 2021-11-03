from Layer import Layer
import numpy as np
from time import time


class NeuralNetwork:
    def __init__(self, eta, input_size):
        self.layers = []
        self.eta = eta
        self.input_size = input_size

    def add_layer(self, layer):
        if isinstance(layer, Layer):
            if self.layers:     # Si la lista de capas no está vacía
                layer.initialize_weights(self.layers[-1].neurons.shape[0])
            else:
                layer.initialize_weights(self.input_size)
            self.layers.append(layer)

    def predict(self, input):
        if input.shape[0] != self.input_size:
            print("Input shape should be", self.input_size, "and is", input.shape)
            return
        self.layers[0].compute_values(input)

        for layer_index in range(len(self.layers)-1):
            self.layers[layer_index+1].compute_values(self.layers[layer_index].activated_neurons)

        return self.layers[-1].activated_neurons

    def fit(self, X, y, epochs, validation_X, validation_Y, batch_size):
        if len(self.layers) < 2:
            return

        history_error = []
        history_accuracy = []
        for epoch in range(epochs):
            print("Epoch", epoch + 1, end=' || ')
            initial_time = time()

            batch_indices = np.random.choice(X.shape[0], batch_size, replace=False)
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]

            for data_index in range(batch_X.shape[0]):
                training_X = batch_X[data_index]
                training_y = batch_y[data_index]
                self.predict(training_X)

                # Calcula el error dada la capa siguiente
                self.layers[-1].compute_error(training_y)
                for i in range(len(self.layers)-2, -1, -1):
                    self.layers[i].compute_error(self.layers[i+1])

                self.layers[0].adjust_weights_and_biases(self.eta, training_X)
                for i in range(1, len(self.layers)):
                    self.layers[i].adjust_weights_and_biases(self.eta, self.layers[i-1])

            error = 0
            correct = 0

            if len(validation_Y.shape) < 2: output_size = 1
            else: output_size = validation_Y[0].shape[0]

            for data_index in range(validation_X.shape[0]):
                prediction = self.predict(validation_X[data_index])
                error += np.sum(np.power(prediction - validation_Y[data_index], 2))/output_size
                correct += (np.round(prediction) == validation_Y[data_index]).all()
            accuracy = 100*(correct/validation_Y.shape[0])
            error /= validation_X.shape[0]
            print("Tiempo (s):", time()-initial_time,
                  "|| Error cuadrático medio:", error,
                  "|| Precisión:", accuracy, "%")
            history_error.append(error)
            history_accuracy.append(accuracy)
        return history_error, history_accuracy
