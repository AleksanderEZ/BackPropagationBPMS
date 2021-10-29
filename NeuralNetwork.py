from Layer import Layer


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
        if input.shape != self.input_size:
            print("Input shape should be", self.input_size, "and is", input.shape)
            return
        self.layers[0].compute_values(input)

        for layer_index in range(len(self.layers)-1):
            self.layers[layer_index+1].compute_values(self.layers[layer_index].activated_neurons)

        return self.layers[-1].activated_neurons

    def fit(self, X, y, epochs):
        if len(self.layers) < 2:
            return
        for epoch in range(epochs):
            print("Epoch", epoch + 1)
            for data_index in range(X.shape[0]):
                training_X = X[data_index]
                training_y = y[data_index]
                self.predict(training_X)

                # Separar en calcular error y ajustar pesos
                self.layers[-1].adjust_weights_and_biases(training_y, self.eta, self.layers[-2])
                for i in range(len(self.layers)-2, 0, -1):
                    self.layers[i].adjust_weights_and_biases(self.eta, self.layers[i-1], self.layers[i+1])
                self.layers[0].adjust_weights_and_biases(self.eta, training_X, self.layers[1])