from dynamic_qcnn.core import Primitive_Types
import warnings
import pennylane as qml


def V(bits, symbols=None):
    qml.CNOT(wires=[bits[0], bits[1]])


def U(bits, symbols=None):
    qml.CRZ(symbols[0], wires=[bits[0], bits[1]])


def convert_graph_to_circuit_pennylane(qcnn, symbols):
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
            for bits in layer.E:
                block(bits=bits, symbols=symbols[coef_indices[ind]])
            ind = ind + 1
        else:
            for bits in layer.E:
                block(bits=bits)
    return coef_indices
