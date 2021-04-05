import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import MNIST


def load_data():
    mnist = MNIST('./mnist', train=True, download=True)
    data = mnist.data
    # normalize data
    data = np.array(data.view(data.shape[0], np.prod(data.shape[1:])))
    data = (data / 255.).astype(np.float32)
    # onehot encode labels
    targets = np.eye(10)[mnist.targets]
    return train_test_split(data, targets, test_size=0.15, random_state=42)


def create_batches(x_data, y_data, batch_size):
    for i in range(0, len(x_data), batch_size):
        x_batch = x_data[i:i + batch_size]
        y_batch = y_data[i:i + batch_size]
        yield(x_batch, y_batch)


def sigmoid(x, derivative=False):
    if derivative:
        return np.exp(-x) / ((np.exp(-x) + 1)**2)
    return 1 / (1 + np.exp(-x))


def softmax(x, derivative=False):
    exponent = np.exp(x)
    if derivative:
        return exponent / np.sum(exponent, axis=0) * \
               (1 - exponent / np.sum(exponent, axis=0))
    return exponent / np.sum(exponent, axis=0)


def mse(x, derivative=False):
    if derivative:
        return 2 * x / x.shape[0]
    return x**2 / x.shape[0]
