"""
Helper functions for cirq
"""
from hierarqcal.core import Primitive_Types
from .cirq_circuits import U2, V2
import warnings
import sympy
import cirq


def get_cirq_default_unitary(layer):
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


def get_circuit_cirq(hierq, symbols=None, pretty=False):
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
    if not (symbols is None):
        # If symbols were provided then set them
        hierq.set_symbols(symbols)
    for layer in hierq:
        # If layer is default mapping we need to set it to cirq default
        if layer.is_default_mapping:
            cirq_default_unitary = get_cirq_default_unitary(layer)
            layer.set_edge_mapping(cirq_default_unitary)
        for unitary in layer.edge_mapping:
            circuit = unitary.function(bits=unitary.edge, symbols=unitary.symbols, circuit=circuit)
    return circuit


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
