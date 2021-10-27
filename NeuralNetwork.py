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
            self.layers[layer_index+1].compute_values(self.layers[layer_index].neurons)

        return self.layers[-1].neurons

    def fit(self, X, y):
        pass