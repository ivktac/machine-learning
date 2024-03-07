import numpy as np
from typing import Callable
from enum import Enum


class ActivationFunction(Enum):
    IDENTITY = "identity"
    STEP = "step"


class Neuron:
    def __init__(
        self,
        weights: np.ndarray,
        bias: float = 0.0,
        threshold: float = 0.0,
        activate_type: ActivationFunction = ActivationFunction.STEP,
    ) -> None:
        self.weights = weights
        self.bias = bias
        self.threshold = threshold
        self.activation_function = self.get_activation_function(activate_type)

    def get_activation_function(self, activate_type: ActivationFunction) -> Callable:
        if activate_type == ActivationFunction.IDENTITY:
            return self.identity
        elif activate_type == ActivationFunction.STEP:
            return self.step
        else:
            raise ValueError(f"Unknown activation function: {activate_type}")

    def activate(self, inputs: np.ndarray) -> float:
        s = np.dot(self.weights, inputs) + self.bias
        return self.activation_function(s)

    @staticmethod
    def identity(x: float) -> float:
        return x

    @staticmethod
    def step(x: float) -> float:
        return 1 if x > 0 else 0


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
