from neural import or_neuron, not_neuron, and_neuron, xor_neuron, Neuron
import numpy as np


def test_logical_and():
    assert and_neuron.activate(np.array([0, 0])) == 0
    assert and_neuron.activate(np.array([0, 1])) == 0
    assert and_neuron.activate(np.array([1, 0])) == 0
    assert and_neuron.activate(np.array([1, 1])) == 1


def test_logical_or():
    assert or_neuron.activate(np.array([0, 0])) == 0
    assert or_neuron.activate(np.array([0, 1])) == 1
    assert or_neuron.activate(np.array([1, 0])) == 1
    assert or_neuron.activate(np.array([1, 1])) == 1


def test_logical_not():
    assert not_neuron.activate(np.array([0])) == 1
    assert not_neuron.activate(np.array([1])) == 0


def test_logical_xor():
    assert xor_neuron.forward_propagate(np.array([0, 0])) == 0
    assert xor_neuron.forward_propagate(np.array([0, 1])) == 1
    assert xor_neuron.forward_propagate(np.array([1, 0])) == 1
    assert xor_neuron.forward_propagate(np.array([1, 1])) == 0


def test_logical_my():
    my_neuron = Neuron(
        weights=np.array([-8.7306697, 4.91139312, 4.37958158]), bias=4.36482319
    )

    assert my_neuron.activate(np.array([0, 0, 0])) == 1
    assert my_neuron.activate(np.array([0, 1, 0])) == 1
    assert my_neuron.activate(np.array([1, 0, 0])) == 0
    assert my_neuron.activate(np.array([1, 1, 1])) == 1
