import numpy as np
from helper import create_batches, load_data
from network import MLP, Layer


def train(model, epochs=10, batch_size=32):
    x_train, x_test, y_train, y_test = load_data()
    for e in range(epochs):
        for batch_x, batch_y in create_batches(x_train, y_train, batch_size):
            model.zero_grad()
            for x, y in zip(batch_x, batch_y):
                model.forward(x, y)
            model.backward()
            model.step()
        print(f"Finished epoch {e}")
        test(model, x_test, y_test)


def test(model, x_test, y_test):
    predictions = []
    for x, y in zip(x_test, y_test):
        output = model.forward(x, y)
        predictions.append(np.argmax(output) == np.argmax(y))
    accuracy = np.mean(predictions)
    print(f"Accuracy: {round(accuracy, 4)}")


if __name__ == "__main__":
    np.random.seed(42)

    model = MLP([
        Layer(28 * 28, 128),
        Layer(128, 64),
        Layer(64, 10)
        ])

    train(model)
