import numpy as np
from typing import Callable


class Neuron:
    def __init__(
        self,
        weights: np.ndarray,
        bias: float = 0.0,
        f: Callable[[float], float] = lambda x: 1 if x > 0 else 0,
    ) -> None:
        self.weights = weights
        self.bias = bias
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
not_neuron = Neuron(weights=np.array([-1.5]), bias=1)
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
