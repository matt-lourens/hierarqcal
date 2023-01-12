from dynamic_qcnn.core import Primitive_Types
import warnings
from qiskit.circuit import Parameter, QuantumCircuit, QuantumRegister

# Default convolution circuit
def U(bits, symbols=None, circuit=None):
    if circuit is None:
        circuit = QuantumCircuit()
    q0, q1 = QuantumRegister(1, f"q{bits[0]}"), QuantumRegister(1, f"q{bits[1]}")
    circuit.crz(symbols[0], q0, q1)
    return circuit


# Default pooling circuit
def V(bits, symbols=None, circuit=None):
    if circuit is None:
        circuit = QuantumCircuit()
    q0, q1 = QuantumRegister(1, f"q{bits[0]}"), QuantumRegister(1, f"q{bits[1]}")
    circuit.cnot(q0, q1)
    return circuit


# def U(bits, symbols=None):
#     circuit = QuantumCircuit(len(bits))
#     circuit.rz(-np.pi / 2, bits[1])
#     circuit.cx(bits[1], bits[0])
#     circuit.rz(symbols[0], bits[0])
#     circuit.ry(symbols[1], bits[1])
#     circuit.cx(bits[0], bits[1])
#     circuit.ry(symbols[2], bits[1])
#     circuit.cx(bits[1], bits[0])
#     circuit.rz(np.pi / 2, bits[0])
#     return circuit


def convert_graph_to_circuit_qiskit(qcnn, pretty=False):
    circuit = QuantumCircuit()
    for q in qcnn.tail.Q:
        circuit.add_register(QuantumRegister(1, f"q{q}"))
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
        required_qs = {f"q{q}" for q in layer.Q_avail}
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
