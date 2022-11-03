import sympy
import cirq

# Default pooling circuit
def V(bits, symbols=None):
    circuit = cirq.Circuit()
    q0, q1 = cirq.LineQubit(bits[0]), cirq.LineQubit(bits[1])
    circuit += cirq.rz(symbols[0]).on(q1).controlled_by(q0)
    circuit += cirq.X(q0)
    circuit += cirq.rx(symbols[1]).on(q1).controlled_by(q0)
    return circuit


# Default convolution circuit
def U(bits, symbols=None):
    circuit = cirq.Circuit()
    q0, q1 = cirq.LineQubit(bits[0]), cirq.LineQubit(bits[1])
    circuit += cirq.rx(symbols[0]).on(q0)
    circuit += cirq.rx(symbols[1]).on(q1)
    circuit += cirq.rz(symbols[2]).on(q0)
    circuit += cirq.rz(symbols[3]).on(q1)
    circuit += cirq.rz(symbols[4]).on(q1).controlled_by(q0)
    circuit += cirq.rz(symbols[5]).on(q0).controlled_by(q1)
    circuit += cirq.rx(symbols[6]).on(q0)
    circuit += cirq.rx(symbols[7]).on(q1)
    circuit += cirq.rz(symbols[8]).on(q0)
    circuit += cirq.rz(symbols[9]).on(q1)
    return circuit


def convert_graph_to_circuit_cirq(qcnn, pretty=False):
    circuit = cirq.Circuit()
    total_coef_count = 0
    symbols = ()
    for layer in qcnn:
        block, block_param_count = layer.function_mapping
        if block_param_count > 0:
            if pretty:
                layer_symbols = sympy.symbols(
                    f"\\theta_{{{total_coef_count}:{total_coef_count + block_param_count}}}"
                )
            else:
                layer_symbols = sympy.symbols(
                    f"x_{total_coef_count}:{total_coef_count + block_param_count}"
                )
            symbols += layer_symbols
            total_coef_count = total_coef_count + block_param_count
        for bits in layer.E:
            if block_param_count > 0:
                circuit.append(block(bits, layer_symbols))
            else:
                # If the circuit has no paramaters then the only argument is bits
                circuit.append(block(bits))
    return circuit, symbols


# def U_s(bits, symbols=None):
#     circuit = cirq.Circuit()
#     q0, q1 = cirq.LineQubit(bits[0]), cirq.LineQubit(bits[1])
#     # circuit += cirq.H(q0)
#     # circuit += cirq.H(q1)
#     circuit += cirq.rz(symbols[0]).on(q1).controlled_by(q0)
#     # circuit += cirq.rz(symbols[1]).on(q0).controlled_by(q1)
#     return circuit


# def V_s(bits, symbols=None):
#     circuit = cirq.Circuit()
#     q0, q1 = cirq.LineQubit(bits[0]), cirq.LineQubit(bits[1])
#     circuit += cirq.CNOT(q0, q1)
#     return circuit
# from core import QConv, QPool, binary_tree_r
# head_graph, _ = binary_tree_r(8, convolution_mapping={1:(U_s,1)}, pooling_mapping={1:(V_s,1)})
# circuit, symbols = convert_graph_to_circuit_cirq(head_graph)



