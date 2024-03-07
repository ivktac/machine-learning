import numpy as np
from typing import Callable


class ArtificalNeuron:
    def __init__(
        self,
        weights: list[float],
        threshold: float = 0.0,
        activation_function: Callable[[float], float] | None = None,
    ):
        self.weights = np.array(weights)
        self.threshold = threshold
        self.activation_function = activation_function or self.linear_activation

    def linear_activation(self, x: float):
        return x

    def compute_weighted_sum(self, inputs: list[float]) -> float:
        return np.dot(inputs, self.weights)

    def activate(self, inputs: list[float]):
        weighted_sum = self.compute_weighted_sum(inputs)
        return (
            self.activation_function(weighted_sum)
            if weighted_sum > self.threshold
            else 0
        )


class NeuralNetwork:
    def __init__(
        self,
        neuron: list[ArtificalNeuron],
        hidden_layer: list[ArtificalNeuron] | None = None,
    ):
        self.output_layer = neuron
        self.hidden_layer = hidden_layer

    def forward_propagation(self, inputs: list[float]) -> list[float]:
        if hidden_layer := self.hidden_layer:
            hidden_layer_outputs = [neuron.activate(inputs) for neuron in hidden_layer]
            return [
                neuron.activate(hidden_layer_outputs) for neuron in self.output_layer
            ]

        return [neuron.activate(inputs) for neuron in self.output_layer]

    def __repr__(self):
        return f"NeuralNetwork(output_layer={self.output_layer}, hidden_layer={self.hidden_layer})"


if __name__ == "__main__":
    logical_and_network = NeuralNetwork(
        neuron=[ArtificalNeuron([1, 1], 1.5, lambda x: 1 if x >= 1.5 else 0)]
    )
    print(logical_and_network.forward_propagation([0, 0]))  # [0]
    print(logical_and_network.forward_propagation([0, 1]))  # [0]
    print(logical_and_network.forward_propagation([1, 0]))  # [0]
    print(logical_and_network.forward_propagation([1, 1]))  # [1]
