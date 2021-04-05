import numpy as np
from helper import sigmoid, softmax, mse


class Layer():
    def __init__(self, input_len, output_len):
        self.input_len = input_len
        self.output_len = output_len
        self.weights = np.random.normal(0, 0.5, (output_len, input_len))
        self.bias = np.random.normal(0, 0.5, output_len)

    def forward(self, x):
        self.input = x
        self.output = self.weights.dot(self.input) + self.bias
        return self.output

    def derivative(self, x):
        return np.dot(self.weights.T, x)


class MLP():
    def __init__(self, layers, learning_rate=0.001):
        self.lr = learning_rate
        self.layers = layers
        self.activations = []
        self.zero_grad()

        for _ in self.layers[:-1]:
            self.activations.append(sigmoid)
        self.activations.append(softmax)

    def forward(self, x, y):
        # forward input through every layer
        for i, (layer, activation) in enumerate(
                    zip(self.layers, self.activations)
                ):
            logits = layer.forward(x)
            x = activation(logits)
            # add up derivatives of activations while propagating forward
            self.d_activations[i] += activation(logits, derivative=True)
        self.loss += mse(x - y, derivative=True)
        return x

    def backward(self):
        # calculate error for last layer
        error = self.loss * self.d_activations[-1]
        self.errors[-1] += error

        # calculate error backward through every layer
        for i in reversed(range(1, len(self.layers))):
            d_layer = self.layers[i].derivative(error)
            error = d_layer * self.d_activations[i - 1]
            self.errors[i - 1] += error

    def step(self):
        # apply gradients to layer weights
        for layer, error in zip(self.layers, self.errors):
            gradient = np.outer(error, layer.input)
            layer.weights -= self.lr * gradient
            layer.bias -= self.lr * error

    def zero_grad(self):
        self.errors = []
        self.d_activations = []

        for layer in self.layers:
            self.d_activations.append(np.zeros(layer.output_len))
            self.errors.append(np.zeros(layer.output_len))

        self.loss = np.zeros(layer.output_len)
