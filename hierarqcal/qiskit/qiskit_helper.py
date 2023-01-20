"""
Helper functions for qiskit
"""
from hierarqcal.core import Primitive_Types
import warnings
from qiskit.circuit import Parameter, QuantumCircuit, QuantumRegister

# Default convolution circuit
def U(bits, symbols=None, circuit=None):
    """
    Default convolution circuit, a simple 2 qubit circuit with a single parameter.

    Args:
        bits (list(string or int)): List of qubit indices/labels, if int then the qubits will be named :code:`f"q{bits[0]}" and f"q{bits[1]}"`
        symbols (tuple(Parameter)): Tuple of symbol values (rotation angles) as a Qiskit Parameter object, can be symbolic or numeric.
        circuit (qiskit.QuantumCircuit): QuantumCircuit object to add operations to, if None then a new QuantumCircuit object will be created.

    Returns:
        circuit (qiskit.QuantumCircuit): QuantumCircuit object
    """
    if circuit is None:
        circuit = QuantumCircuit()
    if type(bits[0]) == int:
        q0, q1 = QuantumRegister(1, f"q{bits[0]}"), QuantumRegister(1, f"q{bits[1]}")
    else:
        # Assume bits are strings and in the correct QASM format
        q0, q1 = QuantumRegister(1, bits[0]), QuantumRegister(1, bits[1])
    circuit.crz(symbols[0], q0, q1)
    return circuit


# Default pooling circuit
def V(bits, symbols=None, circuit=None):
    """
    Default pooling circuit, a simple 2 qubit circuit with no parameters and a controlled controlled operation.

    Args:
        bits (list(string or int)): List of qubit indices/labels, if int then the qubit will be named :code:`f"q{bits[0]}" and f"q{bits[1]}"`
        symbols (tuple(Parameter)): Tuple of symbol values (rotation angles) as a Qiskit Parameter object, can be symbolic or numeric.
        circuit (qiskit.QuantumCircuit): QuantumCircuit object to add operations to, if None then a new QuantumCircuit object will be created.

    Returns:
        circuit (qiskit.QuantumCircuit): QuantumCircuit object
    """
    if circuit is None:
        circuit = QuantumCircuit()
    if type(bits[0]) == int:
        q0, q1 = QuantumRegister(1, f"q{bits[0]}"), QuantumRegister(1, f"q{bits[1]}")
    else:
        # Assume bits are strings and in the correct QASM format
        q0, q1 = QuantumRegister(1, bits[0]), QuantumRegister(1, bits[1])
    circuit.cnot(q0, q1)
    return circuit


def convert_graph_to_circuit_qiskit(qcnn):
    """
    The main helper function for qiskit, it takes a qcnn(:py:class:`hierarqcal.core.Qcnn`) object that describes the cicruit architecture
    and builds a qiskit.QuantumCircuit object with the correct function mappings and symbols.

    If the qubits are provided as ints then the qubit will be named :code:`"q0"` and :code:`"q1"`, otherwise the qubits are assumed to be strings.

    Args:
        qcnn (hierarqcal.core.Qcnn): Qcnn object that describes the circuit architecture, consists of a sequence of motifs (hierarqcal.Qmotif)

    Returns:
        (tuple): Tuple containing:
            * circuit (qiskit.QuantumCircuit): QuantumCircuit object
            * symbols (tuple(Parameter)): Tuple of symbols (rotation angles) as a Qiskit Parameter object.
    """
    circuit = QuantumCircuit()
    for q in qcnn.tail.Q:
        if type(q) == int:
            # If bits were provided as ints then the qubit will be named "q0" and "q1"
            circuit.add_register(QuantumRegister(1, f"q{q}"))
        else:
            # Assume bits are strings and in the correct QASM format
            circuit.add_register(QuantumRegister(1, q))
    total_coef_count = 0
    symbols = ()
    for layer in qcnn:
        # get relevant function mapping
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
        # Check if new qubits were made available
        current_qs = {qr.name for qr in circuit.qregs}
        required_qs = {f"q{q}" if type(q) == int else q for q in layer.Q_avail}
        q_diff = required_qs - current_qs
        if q_diff:
            for q in q_diff:
                circuit.add_register(QuantumRegister(1, q))
        if block_param_count > 0:
            layer_symbols = tuple(
                [
                    Parameter(f"x_{i}")
                    for i in range(
                        total_coef_count, total_coef_count + block_param_count, 1
                    )
                ]
            )
            symbols += layer_symbols
            total_coef_count = total_coef_count + block_param_count
        for bits in layer.E:
            if block_param_count > 0:
                circuit = block(bits, layer_symbols, circuit)
            else:
                # If the circuit has no paramaters then the only argument is bits
                circuit = block(bits, circuit=circuit)
        # Add barrier between layers, except the last one.
        if layer.next:
            circuit.barrier()
    return circuit, symbols
