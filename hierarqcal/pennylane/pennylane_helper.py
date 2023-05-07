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


def V3(bits, symbols=None):
    """
    Default pooling circuit, a simple 2 qubit circuit with no parameters and a controlled controlled operation.

    Args:
        bits (list): List of qubit indices/labels
        symbols (tuple(float)): Tuple of symbol values (rotation angles).
    """
    qml.CNOT(wires=[bits[0], bits[1]])
    qml.CNOT(wires=[bits[2], bits[1]])


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
        if layer.is_default_mapping and layer.mapping == None:
            if layer.type in [
                Primitive_Types.CYCLE.value,
                Primitive_Types.PERMUTE.value,
            ]:
                layer.set_mapping((U, 1))
            elif layer.type in [Primitive_Types.MASK.value]:
                layer.set_mapping((V, 0))
            else:
                warnings.warn(
                    f"No default function mapping for primitive type: {layer.type}, please provide a mapping manually"
                )
        block, block_param_count = layer.mapping
        if block_param_count > 0:
            if len(layer.symbols) > block_param_count:
                # If layer is a sub qcnn
                ranges = tuple()
                current_symbol_count = 0
                for bits in layer.E:
                    ranges += (
                        range(total_coef_count, total_coef_count + block_param_count),
                    )
                    total_coef_count = total_coef_count + block_param_count
                    current_symbol_count += block_param_count
                    if current_symbol_count >= len(layer.symbols):
                        break
                coef_indices[ind] = ranges

            else:
                coef_indices[ind] = range(
                    total_coef_count, total_coef_count + block_param_count
                )
                total_coef_count = total_coef_count + block_param_count
        else:
            coef_indices[ind] = None
        ind = ind + 1
    return total_coef_count, coef_indices


def execute_circuit_pennylane(qcnn, params, coef_indices=None, barriers=True):
    """
    The main helper function for pennylane, it takes a qcnn(:py:class:`hierarqcal.core.Qcnn`) object that describes the cicruit architecture
    and executes the function mappings in the correct order with the correct parameters/symbols.

    Args:
        qcnn (hierarqcal.core.Qcnn): Qcnn object that describes the circuit architecture, consists of a sequence of motifs (:py:class:`hierarqcal.core.Qmotif`)
        params (tuple(float)): Tuple of symbol values (rotation angles)
        coef_indices (dict): Dictionary of indices for each motif, if None, it will be calculated automatically
        barriers (bool): If True, barriers will be inserted between each motif
    """
    ind = 0
    if coef_indices == None:
        total_coef_count, coef_indices = get_param_info_pennylane(qcnn=qcnn)
    for layer in qcnn:
        if layer.is_default_mapping and layer.mapping == None:
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
        block, block_param_count = layer.mapping
        if coef_indices[ind] is None:
            # If layer has no associated params
            for bits in layer.E:
                block(bits=bits, symbols=None)
        else:
            if len(layer.symbols) > block_param_count:
                # If layer is a sub qcnn, same as coef_indices[ind] is a tuple of ranges
                # TODO do this better, I think all coef indices should be tuples
                sub_ind = 0
                for bits in layer.E:
                    block(bits=bits, symbols=params[coef_indices[ind][sub_ind]])
                    sub_ind += 1
                    if sub_ind == len(coef_indices[ind]):
                        # If we have reached the last range, we start from the beginning again
                        sub_ind = 0
            else:
                for bits in layer.E:
                    block(bits=bits, symbols=params[coef_indices[ind]])
        ind = ind + 1
        if barriers:
            qml.Barrier(wires=layer.Q_avail)


# def UU(bits, symbols=None):
#     """
#     Default convolution circuit, a simple 2 qubit circuit with a single parameter.

#     Args:
#         bits (list): List of qubit indices/labels
#         symbols (tuple(float)): Tuple of symbol values (rotation angles).
#     """
#     qml.Identity(wires=[bits[0]])
#     qml.Hadamard(wires=[bits[1]])


# # import pennylane as qml
# from hierarqcal.core import Qcnn, Qfree, Qconv, Qpool, Qdense
# import numpy as np


# # Specify QNode
# dev = qml.device("default.qubit", wires=[i + 1 for i in range(8)])


# @qml.qnode(dev)
# def circuit(motif, params):
#     response_wire = motif.head.Q_avail[-1]  # This is the 'last' available qubit
#     execute_circuit_pennylane(motif, params)  # This executes the compute graph in order
#     return qml.expval(qml.PauliZ(response_wire))


# # Specify drawer
# qml.drawer.use_style("black_white")
# # Get param info
# motif = Qfree(4)+ Qconv(1, mapping=(U, 0))
# total_coef_count, coef_indices = get_param_info_pennylane(motif)
# params = np.random.uniform(0, np.pi, total_coef_count)


# # # Draw
# fig, ax = qml.draw_mpl(circuit)(motif, params)
# # # print("Oppa!")
# # def U2(bits, symbols=None):
# #     """
# #     Default convolution circuit, a simple 2 qubit circuit with no parameters and a controlled operation.

# #     Args:
# #         bits (list): List of qubit indices/labels
# #         symbols (tuple(float)): Tuple of symbol values (rotation angles).
# #     """
# #     U = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
# #     qml.QubitUnitary(np.kron(U, U), wires=[bits[0], bits[1]])


# # def V4(bits, symbols=None):
# #     """
# #     Default pooling circuit, a simple 2 qubit circuit with no parameters and a controlled controlled operation.

# #     Args:
# #         bits (list): List of qubit indices/labels
# #         symbols (tuple(float)): Tuple of symbol values (rotation angles).
# #     """
# #     qml.CNOT(wires=[bits[0], bits[1]])
# #     qml.CNOT(wires=[bits[3], bits[2]])


# from hierarqcal.core import Primitive_Types, Qconv, Qfree, Qpool
# import numpy as np

# nq = 4
# dev = qml.device("default.qubit", wires=[i + 1 for i in range(nq)])


# @qml.qnode(dev)
# def circuit(motif, params):
#     response_wire = motif.head.Q_avail[-1]  # This is the 'last' available qubit
#     execute_circuit_pennylane(motif, params)  # This executes the compute graph in order
#     return qml.expval(qml.PauliZ(response_wire))


# # m1_1 = Qconv(stride=1, step=2, offset=1, boundary="open", mapping=(U, 0))
# # m2_1 = Qpool(stride=1, step=4, filter="1001", mapping=(V, 0), boundary="open", qpu=4)
# # m3_1 = Qconv(1, 2, 0, boundary="open", mapping=(U, 0))
# # m1_2 = m2_1 + m3_1

# motif = Qfree(4) + Qconv(1)

# # # Get param info
# total_coef_count, coef_indices = get_param_info_pennylane(motif)
# params = np.random.uniform(0, np.pi, total_coef_count)

# # Draw
# fig, ax = qml.draw_mpl(circuit)(motif, params)
# # Save to svg
# # fig.savefig(f"/home/matt/Downloads/MERA_GRant.svg", format="svg")

# Change figsisze

# fig.set_size_inches(10, 10)