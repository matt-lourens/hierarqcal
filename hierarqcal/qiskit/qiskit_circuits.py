from qiskit.circuit import Parameter, QuantumCircuit, QuantumRegister


# Default cycle circuit
def U2(bits, symbols=None, circuit=None):
    """
    Default cycle circuit, a simple 2 qubit circuit with a single parameter.

    Args:
        bits (list(string or int)): List of qubit indices/labels, if int then the qubits will be named :code:`f"q{bits[0]}" and f"q{bits[1]}"`
        symbols (tuple(Parameter)): Tuple of symbol values (rotation angles) as a Qiskit Parameter object, can be symbolic or numeric.
        circuit (qiskit.QuantumCircuit): QuantumCircuit object to add operations to, if None then a new QuantumCircuit object will be created.

    Returns:s
        circuit (qiskit.QuantumCircuit): QuantumCircuit object
    """
    # Assume bits are strings and in the correct QASM format
    q0, q1 = QuantumRegister(1, bits[0]), QuantumRegister(1, bits[1])
    circuit.crz(symbols[0], q0, q1)
    return circuit


def U3(bits, symbols=None, circuit=None):
    # Assume bits are strings and in the correct QASM format
    q0, q1, q2 = QuantumRegister(1, bits[0]), QuantumRegister(1, bits[1]), QuantumRegister(1, bits[2])
    circuit.crz(symbols[0], q0, q1)
    circuit.crx(symbols[1], q2, q1)
    return circuit

def H2(bits, symbols=None, circuit=None):
    """
    Example circuit for h_top using the Hadamard gate (H gate).

    Args:
        bits (list(string or int)): List of qubit indices/labels, if int then the qubits will be named :code:`f"q{bits[0]}" and f"q{bits[1]}"`
        symbols (tuple(Parameter)): Tuple of symbol values (rotation angles) as a Qiskit Parameter object, can be symbolic or numeric.
        circuit (qiskit.QuantumCircuit): QuantumCircuit object to add operations to, if None then a new QuantumCircuit object will be created.

    Returns:
        circuit (qiskit.QuantumCircuit): QuantumCircuit object
    """
    # Assume bits are strings and in the correct QASM format
    q0, q1 = QuantumRegister(1, bits[0]), QuantumRegister(1, bits[1])
    circuit.h(q0)
    return circuit

def CR2(bits, symbols=None, circuit=None):
    """
    Example circuit for u_cr using the controlled phase shift gate.

    Args:
        bits (list(string or int)): List of qubit indices/labels, if int then the qubits will be named :code:`f"q{bits[0]}" and f"q{bits[1]}"`
        symbols (tuple(Parameter)): Tuple of symbol values (rotation angles) as a Qiskit Parameter object, can be symbolic or numeric.
        circuit (qiskit.QuantumCircuit): QuantumCircuit object to add operations to, if None then a new QuantumCircuit object will be created.

    Returns:
        circuit (qiskit.QuantumCircuit): QuantumCircuit object
    """
    # Assume bits are strings and in the correct QASM format
    q0, q1 = QuantumRegister(1, bits[0]), QuantumRegister(1, bits[1])
    angle = symbols[0] if symbols else 0.0  # Use a different default symbol if no symbols are provided
    circuit.cp(angle, q0, q1)
    return circuit

# Default mask circuit
def V2(bits, symbols=None, circuit=None):
    """
    Default mask circuit, a simple 2 qubit circuit with no parameters and a controlled operation.

    Args:
        bits (list(string or int)): List of qubit indices/labels, if int then the qubit will be named :code:`f"q{bits[0]}" and f"q{bits[1]}"`
        symbols (tuple(Parameter)): Tuple of symbol values (rotation angles) as a Qiskit Parameter object, can be symbolic or numeric.
        circuit (qiskit.QuantumCircuit): QuantumCircuit object to add operations to, if None then a new QuantumCircuit object will be created.

    Returns:
        circuit (qiskit.QuantumCircuit): QuantumCircuit object
    """
    # Assume bits are strings and in the correct QASM format
    q0, q1 = QuantumRegister(1, bits[0]), QuantumRegister(1, bits[1])
    circuit.cnot(q0, q1)
    return circuit


def V4(bits, symbols=None, circuit=None):
    q0, q1, q2, q3 = (
        QuantumRegister(1, bits[0]),
        QuantumRegister(1, bits[1]),
        QuantumRegister(1, bits[2]),
        QuantumRegister(1, bits[3]),
    )
    circuit.cnot(q0, q1)
    circuit.cnot(q3, q2)
    return circuit
