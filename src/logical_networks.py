import numpy as np
import numpy.typing as npt
from typing import Any, Callable


Number = int | float | bool


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

    def __call__(self, inputs: npt.ArrayLike) -> Any:
        weighted_sum = self._compute_weighted_sum(inputs)
        return self._apply_activation(weighted_sum)

    def _compute_weighted_sum(self, inputs: npt.ArrayLike) -> Number:
        return np.dot(self.weights, inputs) + self.bias

    def _apply_activation(self, weighted_sum: Number) -> Any:
        if self.threshold is None:
            return self.activation_func(weighted_sum)
        return self.activation_func(weighted_sum) if weighted_sum >= self.threshold else 0  # fmt: off


class LogicalGateNetwork:
    def __init__(self, output_neuron: Neuron) -> None:
        self.neuron = output_neuron

    def __call__(self, inputs: npt.ArrayLike) -> Any:
        return self.neuron(inputs)


class LogiclAndNetwork(LogicalGateNetwork):
    def __init__(self) -> None:
        output_neuron = Neuron(
            weights=np.array([1, 1]),
            bias=0,
            activation_func=lambda s: 1 if s >= 0.5 else 0,
            threshold=1.5,
        )
        super().__init__(output_neuron)


class LogicalOrNetwork(LogicalGateNetwork):
    def __init__(self) -> None:
        output_neuron = Neuron(
            weights=np.array([1, 1]),
            bias=0,
            activation_func=lambda s: 1 if s >= 0.5 else 0,
            threshold=0.5,
        )
        super().__init__(output_neuron)


class LogicalNotNetwork(LogicalGateNetwork):
    def __init__(self) -> None:
        output_neuron = Neuron(
            weights=np.array([-1.5]),
            bias=0,
            activation_func=lambda s: 1 if s >= -1 else 0,
            threshold=-1,
        )
        super().__init__(output_neuron)


class LogicalXorNetwork(LogicalGateNetwork):
    _hidden_layer = [
        Neuron(
            weights=np.array([1, -1]),
            bias=0,
            activation_func=lambda s: 1 if s >= 0.5 else 0,
            threshold=0.5,
        ),
        Neuron(
            weights=np.array([-1, 1]),
            bias=0,
            activation_func=lambda s: 1 if s >= 0.5 else 0,
            threshold=0.5,
        ),
    ]

    def __init__(self) -> None:
        output_neuron = Neuron(
            weights=np.array([1, 1]),
            bias=0,
            activation_func=lambda s: 1 if s >= 0.5 else 0,
            threshold=-0.5,
        )
        super().__init__(output_neuron)

    def __call__(self, inputs: npt.ArrayLike):
        return self.neuron([neuron(inputs) for neuron in self._hidden_layer])
