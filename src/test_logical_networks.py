from logical_networks import (
    LogiclAndNetwork,
    LogicalOrNetwork,
    LogicalNotNetwork,
    LogicalXorNetwork,
)


def test_logical_and_network():
    and_network = LogiclAndNetwork()
    assert and_network([0, 0]) == 0
    assert and_network([0, 1]) == 0
    assert and_network([1, 0]) == 0
    assert and_network([1, 1]) == 1


def test_logical_or_network():
    or_network = LogicalOrNetwork()
    assert or_network([0, 0]) == 0
    assert or_network([0, 1]) == 1
    assert or_network([1, 0]) == 1
    assert or_network([1, 1]) == 1


def test_logical_not_network():
    not_network = LogicalNotNetwork()
    assert not_network(0) == 1
    assert not_network(1) == 0


def test_logical_xor_network():
    xor_network = LogicalXorNetwork()
    assert xor_network([0, 0]) == 0
    assert xor_network([0, 1]) == 1
    assert xor_network([1, 0]) == 1
    assert xor_network([1, 1]) == 0
