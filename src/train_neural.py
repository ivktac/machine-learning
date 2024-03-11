import numpy as np


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: float) -> float:
    return x * (1 - x)


class NeuralNetwork:
    def __init__(
        self, input_size: int, output_size: int, learning_rate=0.1, iterations=10_0000
    ) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.random.rand(output_size)

    def feedforward(self, inputs):
        self.hidden_layer_input = np.dot(inputs, self.weights) + self.bias
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)
        return self.hidden_layer_output

    def backpropagation(self, inputs, expected_output):
        error = expected_output - self.hidden_layer_output
        d_predicted_output = error * sigmoid_derivative(self.hidden_layer_output)

        self.weights += np.dot(inputs.T, d_predicted_output) * self.learning_rate
        self.bias += np.sum(d_predicted_output) * self.learning_rate

        return error

    def train(self, inputs, expected_output):
        for _ in range(self.iterations):
            self.feedforward(inputs)
            error = self.backpropagation(inputs, expected_output)

            if np.all(np.abs(error) < 1e-5):
                break

    def predict(self, inputs):
        return self.feedforward(inputs)


if __name__ == "__main__":
    inputs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
    expected_output = np.array([[1], [1], [0], [1]])

    neural_network = NeuralNetwork(input_size=3, output_size=1)
    neural_network.train(inputs, expected_output)

    print("Тестування нейронної мережі:")
    for input_data in inputs:
        prediction = neural_network.predict(input_data)
        print(f"{input_data} -> {prediction}")

    print("Ваги та зсув:")
    print(neural_network.weights, neural_network.bias)
