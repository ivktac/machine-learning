import numpy as np
import numpy.typing as npt
from typing import Callable


class Neuron:
    def __init__(
        self,
        weights: npt.ArrayLike,
        bias: float,
        activation_func: Callable[[float], float],
        threshold: float | None = None,
    ):
        self.weights = weights
        self.bias = bias
        self.activation_func = activation_func
        self.threshold = threshold

    def __call__(self, inputs: npt.ArrayLike):
        weighted_sum = np.dot(self.weights, inputs) + self.bias

        return (
            self.activation_func(weighted_sum)
            if self.threshold is None
            else self.activation_func(weighted_sum)
            if weighted_sum >= self.threshold
            else 0
        )


class LogiclAndNetwork:
    def __init__(self) -> None:
        self.neuron = Neuron(
            weights=np.array([1, 1]),
            bias=-0.5,
            activation_func=lambda s: 1 if s >= 1.5 else 0,
            threshold=1.5,
        )

    def __call__(self, inputs: npt.ArrayLike):
        return self.neuron(inputs)


class LogicalOrNetwork:
    def __init__(self) -> None:
        self.neuron = Neuron(
            weights=np.array([1, 1]),
            bias=0,
            activation_func=lambda s: 1 if s >= 0.5 else 0,
            threshold=0.5,
        )

    def __call__(self, inputs: npt.ArrayLike):
        return self.neuron(inputs)


class LogicalNotNetwork:
    def __init__(self) -> None:
        self.neuron = Neuron(
            weights=np.array([-1.5]),
            bias=0,
            activation_func=lambda s: 1 if s >= -1 else 0,
            threshold=-1,
        )

    def __call__(self, input: bool | int):
        return self.neuron(np.array([input]))


class LogicalXorNetwork:
    def __init__(self) -> None:
        self.hidden_layer = [
            Neuron(
                weights=np.array([1, -1]),
                bias=0,
                activation_func=lambda s: 1 if s >= 0.5 else 0,
                threshold=-0.5,
            ),
            Neuron(
                weights=np.array([-1, 1]),
                bias=0,
                activation_func=lambda s: 1 if s >= 0.5 else 0,
                threshold=-0.5,
            ),
        ]

        self.neuron = Neuron(
            weights=np.array([1, 1]),
            bias=0,
            activation_func=lambda s: 1 if s >= 0.5 else 0,
            threshold=-0.5,
        )

    def __call__(self, inputs: npt.ArrayLike):
        return self.neuron([neuron(inputs) for neuron in self.hidden_layer])
