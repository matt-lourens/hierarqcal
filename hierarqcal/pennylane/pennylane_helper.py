"""
Helper functions for cirq
"""
import warnings
from sympy import symbols, lambdify
import pennylane as qml
from .pennylane_circuits import U2, V2
from hierarqcal.core import Primitive_Types

def get_pennylane_default_unitary(layer):
    if layer.type in [
        Primitive_Types.CYCLE,
        Primitive_Types.PERMUTE,
    ]:
        unitary_function = U2
    elif layer.type in [Primitive_Types.MASK]:
        unitary_function = V2
    else:
        warnings.warn(
            f"No default function mapping for primitive type: {layer.type}, please provide a mapping manually"
        )
    # Give all edge mappings correct default unitary
    return unitary_function


def execute_circuit_pennylane(hierq, symbols=None, barriers=True):
    """
    The main helper function for pennylane, it takes a qcnn(:py:class:`hierarqcal.core.Qcnn`) object that describes the cicruit architecture
    and executes the function mappings in the correct order with the correct parameters/symbols.

    Args:
        qcnn (hierarqcal.core.Qcnn): Qcnn object that describes the circuit architecture, consists of a sequence of motifs (:py:class:`hierarqcal.core.Qmotif`)
        params (tuple(float)): Tuple of symbol values (rotation angles)
        coef_indices (dict): Dictionary of indices for each motif, if None, it will be calculated automatically
        barriers (bool): If True, barriers will be inserted between each motif
    """
    if not (symbols is None):
        hierq.set_symbols(symbols)
    for layer in hierq:
        # If layer is default mapping we need to set it to pennylane default
        if layer.is_default_mapping:
            pennylane_default_unitary = get_pennylane_default_unitary(layer)
            layer.set_edge_mapping(pennylane_default_unitary)
        for unitary in layer.edge_mapping:
            unitary.function(bits=unitary.edge, symbols=unitary.symbols)
        if barriers:
            qml.Barrier(wires=hierq.tail.Q_avail)


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
