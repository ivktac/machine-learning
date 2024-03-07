import numpy as np


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


class TimeSeriesPredictor:
    def __init__(self, input_dimension=3, learning_rate=0.01, iterations=10_000):
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = np.random.rand(input_dimension)

    def predict(self, inputs: np.ndarray):
        weighted_sum = np.dot(inputs, self.weights)
        return sigmoid(weighted_sum) * 10

    def train(self, training_inputs: np.ndarray, training_outputs: np.ndarray):
        for _ in range(self.iterations):
            for inputs, target in zip(training_inputs, training_outputs):
                prediction = self.predict(inputs)
                error = target - prediction
                adjustments = self.learning_rate * error * inputs
                self.weights += adjustments


if __name__ == "__main__":
    initial_data = np.array([2.54, 5.28, 0.78, 5.72, 0.58, 4.65, 0.91, 5.80, 1.76, 5.67, 1.73, 5.70, 1.03, 5.00, 1.79])  # fmt: off
    data = np.array(initial_data).reshape(-1, 3)
    outputs = np.array(initial_data[3::3])

    predictor = TimeSeriesPredictor()
    predictor.train(data, outputs)

    test_data = [np.array([0.26, 4.21, 1.90]), np.array([4.21, 1.90, 4.08])]
    actual_data = [5.00, 1.79]
    for i, test_input in enumerate(test_data):
        prediction = predictor.predict(test_input)
        print(
            f"test data: {test_input} -> prediction: {prediction} -> actual: {actual_data[i]}"
        )
