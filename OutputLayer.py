from Layer import Layer
import numpy as np


class OutputLayer(Layer):
    def compute_error(self, y):
        self.error = (self.activated_neurons - y)*(self.activated_neurons*(1-self.activated_neurons))

    def actual_error(self, y):
        return (self.activated_neurons - y)*(self.activated_neurons - y)

