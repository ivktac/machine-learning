import numpy as np
from typing import Callable


class Neuron:
    def __init__(
        self,
        weights: np.ndarray,
        bias: float = 0.0,
        threshold: float = 0.0,
        f: Callable[[float], float] = lambda x: 1 if x > 0 else 0,
    ) -> None:
        self.weights = weights
        self.bias = bias
        self.threshold = threshold
        self.activation_function = f

    def activate(self, inputs: np.ndarray) -> float:
        s = np.dot(self.weights, inputs) + self.bias
        return self.activation_function(s)


class Layer:
    def __init__(self, neurons: list[Neuron] = []) -> None:
        self.neurons = neurons

    def add_neuron(self, neuron: Neuron) -> None:
        self.neurons.append(neuron)

    def activate(self, inputs: np.ndarray) -> np.ndarray:
        return np.array([neuron.activate(inputs) for neuron in self.neurons])


class NeuralNetwork:
    def __init__(self, output_neuron: Neuron, layers: list[Layer] = []) -> None:
        self.output_neuron = output_neuron
        self.layers = layers

    def add_layer(self, layer: Layer) -> None:
        self.layers.append(layer)

    def forward_propagate(self, inputs: np.ndarray) -> float:
        for layer in self.layers:
            inputs = layer.activate(inputs)
        return self.output_neuron.activate(inputs)


or_neuron = Neuron(weights=np.array([1, 1]), bias=-0.5)
not_neuron = Neuron(weights=np.array([-1]), bias=0.5)
and_neuron = Neuron(weights=np.array([1, 1]), bias=-1.5)
xor_neuron = NeuralNetwork(
    output_neuron=Neuron(weights=np.array([1, 1]), bias=-0.5),
    layers=[
        Layer(
            neurons=[
                Neuron(weights=np.array([1, -1]), bias=-0.5),
                Neuron(weights=np.array([-1, 1]), bias=-0.5),
            ]
        )
    ],
)
my_neuron = NeuralNetwork(
    output_neuron=Neuron(
        weights=np.array([-6, 4, 3]), bias=3.1, f=lambda x: 1 if x > 0.5 else 0
    )
)


class SingleLayerPerception:
    def __init__(self, input_size: int, learning_rate=0.01):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate

    def perdict(self, inputs):
        return np.dot(inputs, self.weights) + self.bias

    def train(self, training_inputs, training_labels, epochs=10000):
        for _ in range(epochs):
            for inputs, label in zip(training_inputs, training_labels):
                prediction = self.perdict(inputs)
                error = prediction - label
                self.weights -= self.learning_rate * error * inputs
                self.bias -= self.learning_rate * error


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

    perception = SingleLayerPerception(input_size=3)
    perception.train(X_train, y_train)

    predictions = [perception.perdict(x) for x in X_test]

    print(f"Predicted values: {predictions}")
    print(f"Actual values: {y_test}")
