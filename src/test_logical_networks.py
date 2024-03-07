from neural import NeuralNetwork, ArtificalNeuron


def test_logical_and_network():
    logical_and_network = NeuralNetwork(
        neuron=[
            ArtificalNeuron(
                weights=[1, 1],
                threshold=1.5,
                activation_function=lambda x: 1 if x >= 1.5 else 0,
            )
        ]
    )

    assert logical_and_network.forward_propagation([0, 0]) == [0]
    assert logical_and_network.forward_propagation([0, 1]) == [0]
    assert logical_and_network.forward_propagation([1, 0]) == [0]
    assert logical_and_network.forward_propagation([1, 1]) == [1]


def test_logical_or_network():
    logical_or_network = NeuralNetwork(
        neuron=[
            ArtificalNeuron(
                weights=[1, 1],
                threshold=0.5,
                activation_function=lambda x: 1 if x >= 0.5 else 0,
            )
        ]
    )

    assert logical_or_network.forward_propagation([0, 0]) == [0]
    assert logical_or_network.forward_propagation([0, 1]) == [1]
    assert logical_or_network.forward_propagation([1, 0]) == [1]
    assert logical_or_network.forward_propagation([1, 1]) == [1]


def test_logical_not_network():
    logical_not_network = NeuralNetwork(
        neuron=[
            ArtificalNeuron(
                weights=[-1.5],
                threshold=-1,
                activation_function=lambda x: 1 if x >= -1 else 0,
            )
        ]
    )

    assert logical_not_network.forward_propagation([0]) == [1]
    assert logical_not_network.forward_propagation([1]) == [0]


class LogicalXorNetwork(NeuralNetwork):
    def __init__(self):
        hidden_layer = [
            ArtificalNeuron(
                weights=[1, -1],
                threshold=-0.5,
                activation_function=lambda x: 1 if x >= 0.5 else 0,
            ),
            ArtificalNeuron(
                weights=[-1, 1],
                threshold=-0.5,
                activation_function=lambda x: 1 if x >= 0.5 else 0,
            ),
        ]
        output_neuron = ArtificalNeuron(
            weights=[1, 1],
            threshold=-0.5,
            activation_function=lambda x: 1 if x >= 0.5 else 0,
        )
        super().__init__(neuron=[output_neuron], hidden_layer=hidden_layer)


def test_logical_xor_network():
    xor_network = LogicalXorNetwork()

    assert xor_network.forward_propagation([0, 0]) == [0]
    assert xor_network.forward_propagation([0, 1]) == [1]
    assert xor_network.forward_propagation([1, 0]) == [1]
    assert xor_network.forward_propagation([1, 1]) == [0]


class LogicalMyOpNetwork(NeuralNetwork):
    def __init__(self):
        pass
