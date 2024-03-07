import numpy as np


class SingleLayerPerceptron:
    def __init__(
        self,
        size: int,
        learning_rate=0.01,
    ):
        self.weights = np.random.rand(size)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return weighted_sum

    def fit(self, input_data, expected_data, epochs=10_000):
        for _ in range(epochs):
            total_error = 0

            for inputs, label in zip(input_data, expected_data):
                prediction = self.predict(inputs)
                error = prediction - label

                gradients = error * inputs
                bias_gradient = error

                self.weights -= self.learning_rate * gradients
                self.bias -= self.learning_rate * bias_gradient

                total_error += error**2
                if total_error < 1e-5:
                    break


if __name__ == "__main__":
    data = np.array([2.54, 5.28, 0.78, 5.72, 0.58, 4.65, 0.91, 5.80, 1.76, 5.67, 1.73, 5.70, 1.03, 5.00, 1.79])  # fmt: off
    X = []
    y = []

    for i in range(len(data) - 3):
        X.append(data[i : i + 3])
        y.append(data[i + 3])

    X_train = np.array(X[:10])
    y_train = np.array(y[:10])

    X_test = np.array(X[10:])
    y_test = np.array(y[10:])

    perception = SingleLayerPerceptron(size=X_train.shape[1])
    perception.fit(X_train, y_train)

    predictions = perception.predict(X_test)

    print(f"Predicted values: {predictions}")
    print(f"Actual values: {y_test}")
