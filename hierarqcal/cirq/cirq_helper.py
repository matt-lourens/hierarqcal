"""
Helper functions for cirq
"""
from hierarqcal.core import Primitive_Types
import warnings
import sympy
import cirq


# Default conviolution circuit
def U(bits, symbols=None):
    """
    Default convolution circuit, a simple 2 qubit circuit with a single parameter.

    Args:
        bits (list): List of qubit indices/labels
        symbols (tuple(sympy.Symbol or float)): Tuple of symbols (rotation angles), can be symbolic (sympy) or numeric

    Returns:
        circuit (cirq.Circuit): cirq.Circuit object
    """
    q0, q1 = cirq.LineQubit(bits[0]), cirq.LineQubit(bits[1])
    circuit = cirq.Circuit()
    circuit += cirq.rz(symbols[0]).on(q1).controlled_by(q0)
    return circuit


# Default 3 qubit convolution circuit
def U3(bits, symbols=None):
    """
    Default convolution circuit, 3 qubit gate: Toffoli with the control qubits in the X basis

    Args:
        bits (list): List of qubit indices/labels
        symbols (tuple(sympy.Symbol or float)): Tuple of symbols (rotation angles), can be symbolic (sympy) or numeric

    Returns:
        circuit (cirq.Circuit): cirq.Circuit object
    """
    q0, q1, q2 = (
        cirq.LineQubit(bits[0]),
        cirq.LineQubit(bits[1]),
        cirq.LineQubit(bits[2]),
    )
    circuit = cirq.Circuit()
    circuit += cirq.CCNOT(q0, q2, q1)

    return circuit


# Default pooling circuit
def V(bits, symbols=None):
    """
    Default pooling circuit, a simple 2 qubit circuit with no parameters and a controlled controlled operation.

    Args:
        bits (list): List of qubit indices/labels
        symbols (tuple(sympy.Symbol or float)): Tuple of symbols (rotation angles), can be symbolic (sympy) or numeric

    Returns:
        circuit (cirq.Circuit): cirq.Circuit object
    """
    circuit = cirq.Circuit()
    q0, q1 = cirq.LineQubit(bits[0]), cirq.LineQubit(bits[1])
    circuit += cirq.CNOT(q0, q1)
    return circuit


# # Default pooling circuit
# def V3(bits, symbols=None):
#     """
#     Default pooling circuit, a simple 2 qubit circuit with no parameters and a controlled controlled operation.

#     Args:
#         bits (list): List of qubit indices/labels
#         symbols (tuple(sympy.Symbol or float)): Tuple of symbols (rotation angles), can be symbolic (sympy) or numeric

#     Returns:
#         circuit (cirq.Circuit): cirq.Circuit object
#     """
#     circuit = cirq.Circuit()
#     q0, q1, q2 = (
#         cirq.LineQubit(bits[0]),
#         cirq.LineQubit(bits[1]),
#         cirq.LineQubit(bits[2]),
#     )
#     circuit += cirq.CZ(q0, q1)
#     circuit += cirq.CZ(q2, q1)
#     return circuit


def convert_graph_to_circuit_cirq(qcnn, pretty=False):
    """
    The main helper function for cirq, it takes a qcnn (:py:class:`hierarqcal.core.Qcnn`) object that describes the cicruit architecture
    and converts it to a cirq.Circuit object by connecting the symbols and execution order of function mappings. Essentially goes through
    each operational motif of the qcnn and executes it's function mapping with the correct parameter(symbol) values.

    Args:
        qcnn (hierarqcal.core.Qcnn): Qcnn object that describes the circuit architecture, consists of a sequence of motifs (hierarqcal.Qmotif)
        pretty (bool): If True then the symbols will be formatted as pretty latex symbols, otherwise they will be formatted as x_0, x_1, x_2, ...

    Returns:
        (tuple): Tuple containing:
            * circuit (cirq.Circuit): cirq.Circuit object
            * symbols (tuple(sympy.Symbol or float)): Tuple of symbols (rotation angles).
    """
    circuit = cirq.Circuit()
    symbols = ()
    for layer in qcnn:
        layer_coef_count = 0
        if layer.is_default_mapping and layer.mapping == None:
            type_check = layer.sub_type if layer.sub_type else layer.type
            if type_check in [
                Primitive_Types.CONVOLUTION.value,
                Primitive_Types.DENSE.value,
            ]:
                if layer.qpu == 3:
                    layer.set_mapping((U3, 1))
                else:
                    layer.set_mapping((U, 1))
            elif type_check in [Primitive_Types.POOLING.value]:
                layer.set_mapping((V, 0))
            else:
                warnings.warn(
                    f"No default function mapping for primitive type: {type_check}, please provide a mapping manually"
                )
        block, block_param_count = layer.mapping
        for bits in layer.E:
            if block_param_count > 0:
                layer_symbols = layer.symbols[
                    layer_coef_count : layer_coef_count + block_param_count
                ]
                if len(layer.symbols) > layer_coef_count + block_param_count:
                    layer_coef_count = layer_coef_count + block_param_count
                else:
                    layer_coef_count = 0
                circuit.append(block(bits, layer_symbols))
            else:
                # If the circuit has no parameters then the only argument is bits
                circuit.append(block(bits))
        # if layer.share_weights == True:
        #     if block_param_count > 0:
        #         if pretty:
        #             layer_symbols = sympy.symbols(
        #                 f"\\theta_{{{total_coef_count}:{total_coef_count + block_param_count}}}"
        #             )
        #         else:
        #             layer_symbols = sympy.symbols(
        #                 f"x_{total_coef_count}:{total_coef_count + block_param_count}"
        #             )
        #         symbols += layer_symbols
        #         total_coef_count = total_coef_count + block_param_count
        #     for bits in layer.E:
        #         if block_param_count > 0:
        #             circuit.append(block(bits, layer_symbols))
        #         else:
        #             # If the circuit has no parameters then the only argument is bits
        #             circuit.append(block(bits))
        # else:
        #     # If the layer does not share weights then we need to create a new set of symbols for each block
        #     for bits in layer.E:
        #         if block_param_count > 0:
        #             layer_symbols = sympy.symbols(
        #                 f"x_{total_coef_count}:{total_coef_count + block_param_count}"
        #             )
        #             symbols += layer_symbols
        #             total_coef_count = total_coef_count + block_param_count
        #             circuit.append(block(bits, layer_symbols))
        #         else:
        #             # If the circuit has no paramaters then the only argument is bits
        #             circuit.append(block(bits))
    return circuit, qcnn.symbols


def _pretty_cirq_plot(circuit, out):
    import cirq.contrib.qcircuit as ccq

    # from cirq.contrib.svg import SVGCircuit
    # SVGCircuit(circuit)
    a = ccq.circuit_to_latex_using_qcircuit(circuit)
    with open(
        f"{out}",
        "a",
    ) as f:
        f.write(f"\\newline\n" f"{a}\\newline\n")


# Cool idea but will have to be implementation specific.
# def lambdify(motif):
#     circuit, symbols = convert_graph_to_circuit_cirq(motif)
#     module_dictionary = {'child_circuit': circuit, 'child_symbols': symbols}
#     exec('''
#         def U(bits, symbols=None):
#             circuit = cirq.Circuit()
#             total_coef_count = 0
#             for layer in qcnn:
#                 block, block_param_count = layer.mapping
#                 # If the layer does not share weights then we need to create a new set of symbols for each block
#                 for bits in layer.E:
#                     if block_param_count > 0:
#                         layer_symbols = symbols[total_coef_count:total_coef_count + block_param_count]
#                         total_coef_count = total_coef_count + block_param_count
#                         circuit.append(block(bits, layer_symbols))
#                     else:
#                         # If the circuit has no parameters then the only argument is bits
#                         circuit.append(block(bits))
#             return circuit
#     ''')

# from hierarqcal import Qcnn, Qfree, Qconv, Qpool, Qdense, plot_motif, plot_motifs

# m = Qfree(3) + Qdense(permutations=True, share_weights=True)
# # m = Qfree(8) + Qpool(filter="right")
# m1 = Qfree(16) + Qconv(1, 3, 0, mapping=m, share_weights=False)
# circuit, symbols = convert_graph_to_circuit_cirq(m1)
# print(circuit)
# print("Oi")
