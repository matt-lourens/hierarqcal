import cirq

# Default cycle circuit
def U2(bits, symbols=None, circuit=None):
    """
    Default cycle circuit, a simple 2 qubit circuit with a single parameter.

    Args:
        bits (list): List of qubit indices/labels
        symbols (tuple(sympy.Symbol or float)): Tuple of symbols (rotation angles), can be symbolic (sympy) or numeric

    Returns:
        circuit (cirq.Circuit): cirq.Circuit object
    """
    q0, q1 = cirq.LineQubit(bits[0]), cirq.LineQubit(bits[1])
    circuit += cirq.rz(symbols[0]).on(q1).controlled_by(q0)
    return circuit


# Default 3 qubit cycle circuit
def U3(bits, symbols=None, circuit=None):
    """
    Default cycle circuit, 3 qubit gate: Toffoli with the control qubits in the X basis

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
    circuit += cirq.CCNOT(q0, q2, q1)

    return circuit


# Default mask circuit
def V2(bits, symbols=None, circuit=None):
    """
    Default mask circuit, a simple 2 qubit circuit with no parameters and a controlled controlled operation.

    Args:
        bits (list): List of qubit indices/labels
        symbols (tuple(sympy.Symbol or float)): Tuple of symbols (rotation angles), can be symbolic (sympy) or numeric

    Returns:
        circuit (cirq.Circuit): cirq.Circuit object
    """
    q0, q1 = cirq.LineQubit(bits[0]), cirq.LineQubit(bits[1])
    circuit += cirq.CNOT(q0, q1)
    return circuit

def V4(bits, symbols=None, circuit=None):
    q0, q1, q2, q3 = (
        cirq.LineQubit(bits[0]),
        cirq.LineQubit(bits[1]),
        cirq.LineQubit(bits[2]),
        cirq.LineQubit(bits[3]),
    )
    circuit += cirq.CNOT(q0, q1)
    circuit += cirq.CNOT(q3, q2)
    return circuit