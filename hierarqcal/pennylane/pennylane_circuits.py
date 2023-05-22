import pennylane as qml
from hierarqcal.core import Qunitary


def V2(bits, symbols=None):
    """
    Default pooling circuit, a simple 2 qubit circuit with no parameters and a controlled controlled operation.

    Args:
        bits (list): List of qubit indices/labels
        symbols (tuple(float)): Tuple of symbol values (rotation angles).
    """
    qml.CNOT(wires=[bits[0], bits[1]])


def V3(bits, symbols=None):
    """
    Default pooling circuit, a simple 2 qubit circuit with no parameters and a controlled controlled operation.

    Args:
        bits (list): List of qubit indices/labels
        symbols (tuple(float)): Tuple of symbol values (rotation angles).
    """
    qml.CNOT(wires=[bits[0], bits[1]])
    qml.CNOT(wires=[bits[2], bits[1]])


def V4(bits, symbols=None):
    """
    Default pooling circuit, a simple 2 qubit circuit with no parameters and a controlled controlled operation. TODO docstring

    Args:
        bits (list): List of qubit indices/labels
        symbols (tuple(float)): Tuple of symbol values (rotation angles).
    """
    qml.CNOT(wires=[bits[0], bits[1]])
    qml.CNOT(wires=[bits[3], bits[2]])


def U2(bits, symbols=None):
    """
    Default convolution circuit, a simple 2 qubit circuit with a single parameter.

    Args:
        bits (list): List of qubit indices/labels
        symbols (tuple(float)): Tuple of symbol values (rotation angles).
    """
    qml.CRZ(symbols[0], wires=[bits[0], bits[1]])
