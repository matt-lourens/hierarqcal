"""
Helper functions for cirq
"""
import warnings
from sympy import symbols, lambdify
import pennylane as qml
from hierarqcal.core import Primitive_Types


def V(bits, symbols=None):
    """
    Default pooling circuit, a simple 2 qubit circuit with no parameters and a controlled controlled operation.

    Args:
        bits (list): List of qubit indices/labels
        symbols (tuple(float)): Tuple of symbol values (rotation angles).
    """
    qml.CNOT(wires=[bits[0], bits[1]])


def U(bits, symbols=None):
    """
    Default convolution circuit, a simple 2 qubit circuit with a single parameter.

    Args:
        bits (list): List of qubit indices/labels
        symbols (tuple(float)): Tuple of symbol values (rotation angles).
    """
    qml.CRZ(symbols[0], wires=[bits[0], bits[1]])


def get_param_info_pennylane(qcnn):
    """
    Helper function that returns the total number of parameters and a dictionary that maps the parameter indices to the motifs (in the order they occur).

    Args:
        qcnn (hierarqcal.core.Qcnn): Qcnn object that describes the circuit architecture, consists of a sequence of motifs (hierarqcal.core.Qmotif)

    Returns:
        (tuple): Tuple containing:

            * total_coef_count (int): Total number of parameters.
            * coef_indices (dict): Dictionary that maps the parameter indices to the motifs (in the order they occur).

    """
    total_coef_count = 0
    coef_indices = {}
    ind = 0
    for layer in qcnn:
        if layer.is_default_mapping and layer.function_mapping == None:
            if layer.type in [
                Primitive_Types.CONVOLUTION.value,
                Primitive_Types.DENSE.value,
            ]:
                layer.set_mapping((U, 1))
            elif layer.type in [Primitive_Types.POOLING.value]:
                layer.set_mapping((V, 0))
            else:
                warnings.warn(
                    f"No default function mapping for primitive type: {layer.type}, please provide a mapping manually"
                )
        block, block_param_count = layer.function_mapping
        if block_param_count > 0:
            coef_indices[ind] = range(
                total_coef_count, total_coef_count + block_param_count
            )
            total_coef_count = total_coef_count + block_param_count
        else:
            coef_indices[ind] = None
        ind = ind + 1
    return total_coef_count, coef_indices


def execute_circuit_pennylane(qcnn, params, coef_indices=None):
    """
    The main helper function for pennylane, it takes a qcnn(:py:class:`hierarqcal.core.Qcnn`) object that describes the cicruit architecture
    and executes the function mappings in the correct order with the correct parameters/symbols.

    Args:
        qcnn (hierarqcal.core.Qcnn): Qcnn object that describes the circuit architecture, consists of a sequence of motifs (:py:class:`hierarqcal.core.Qmotif`)
        params (tuple(float)): Tuple of symbol values (rotation angles)
        coef_indices (dict): Dictionary of indices for each motif, if None, it will be calculated automatically
    """
    ind = 0
    if coef_indices == None:
        total_coef_count, coef_indices = get_param_info_pennylane(qcnn=qcnn)
    for layer in qcnn:
        if layer.is_default_mapping and layer.function_mapping == None:
            if layer.type in [
                Primitive_Types.CONVOLUTION.value,
                Primitive_Types.DENSE.value,
            ]:
                layer.set_mapping((U, 1))
            elif layer.type in [Primitive_Types.POOLING.value]:
                layer.set_mapping((V, 0))
            else:
                warnings.warn(
                    f"No default function mapping for primitive type: {layer.type}, please provide a mapping manually"
                )
        block, block_param_count = layer.function_mapping
        if coef_indices[ind] is None:
            # If layer has no associated params
            for bits in layer.E:
                block(bits=bits, symbols=None)
        else:
            for bits in layer.E:
                block(bits=bits, symbols=params[coef_indices[ind]])
        ind = ind + 1
