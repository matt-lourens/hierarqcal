import warnings
from sympy import symbols
import pennylane as qml
from dynamic_qcnn.core import Primitive_Types



def V(bits, symbols=None):
    qml.CNOT(wires=[bits[0], bits[1]])


def U(bits, symbols=None):
    qml.CRZ(symbols[0], wires=[bits[0], bits[1]])


def get_param_info_pennylane(qcnn):
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
        symbols_coef = symbols(f'x:{total_coef_count}')
    return symbols_coef, total_coef_count, coef_indices


def execute_circuit_pennylane(qcnn, symbols, coef_indices=None):
    ind = 0
    if coef_indices == None:
        coef_indices = get_symbol_indices_pennylane(qcnn=qcnn)
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
        for bits in layer.E:
            block(bits=bits, symbols=symbols[coef_indices[ind]])
        ind = ind + 1
