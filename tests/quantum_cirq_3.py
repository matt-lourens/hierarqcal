# %%
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
from tensorflow import keras

# %%
class Qubit_Encoding(keras.layers.Layer):
    def __init__(self, gate=cirq.rx, encoding_args={}, name="qubit_encoding", **kwargs):
        super(Qubit_Encoding, self).__init__(name=name, **kwargs)
        self.gate = gate

    def call(self, inputs):
        circuit = cirq.Circuit()
        for i, value in enumerate(inputs):
            qubit = cirq.LineQubit(i)
            circuit.append(self.gate(inputs[i]).on(qubit))
        return circuit


class Block:
    """
    A generic circuit block consisting of some combination of variational circuits.
    Order doesn't have to be from 0, all layers get sorted purely by order value
    """

    def __init__(
        self, unit_fn, param_count, wire_pattern, block_order, predefined_structure=None
    ):
        self.unit_fn = unit_fn
        self.param_count = param_count
        self.wire_pattern = wire_pattern
        self.block_order = block_order
        self.predefined_structure = predefined_structure


class Qcnn_Classifier(keras.layers.Layer):
    def __init__(
        self,
        s_c=1,
        s_p=0,
        pool_filter="right",
        convolution_mapping={1: None},
        pooling_mapping={1: None},
        name="qcnn_classifier",
        **kwargs,
    ):
        super(Qcnn_Classifier, self).__init__(name=name, **kwargs)
        self.s_c = s_c
        self.s_p = s_p
        self.pool_filter = pool_filter
        self.convolution_mapping = convolution_mapping
        self.pooling_mapping = pooling_mapping

    def call(self, inputs):
        self.qcnn_graphs_ = self._get_qcnn_graphs(inputs.shape[-1])
        self.circuit = self._construct_circuit()
        return self.circuit

    def _construct_circuit(self):
        circuit = cirq.Circuit()
        total_coef_count = 0
        for layer, graph in self.qcnn_graphs_.items():
            # Notational scheme is layer -> C graph --> Qc,Ec, P Graph --> Qp,Ep
            E_cl = graph[layer][0][1]
            E_pl = graph[layer][1][1]
            convolution_block, c_param_count = self.convolution_mapping.get(layer, U)
            pooling_block, p_parm_count = self.pooling_mapping.get(layer, V)
            # Convolution Operation
            for bits in E_cl:
                symbols = sympy.symbols(
                    f"x{total_coef_count}:{total_coef_count + c_param_count}"
                )
                circuit.append(convolution_block(bits, symbols))
            total_coef_count = total_coef_count + c_param_count
            # Pooling Operation
            for bits in E_pl:
                symbols = sympy.symbols(
                    f"x{total_coef_count}:{total_coef_count + p_parm_count}"
                )
                circuit.append(pooling_block(bits, symbols))
        return circuit

    def _get_qcnn_graphs(self, n_wires):
        """ """
        if type(self.pool_filter) is str:
            # Mapping words to the filter type
            if self.pool_filter == "left":
                # 0 1 2 3 4 5 6 7
                # x x x x
                self.pool_filter = lambda arr: arr[0 : len(arr) // 2 : 1]
            elif self.pool_filter == "right":
                # 0 1 2 3 4 5 6 7
                #         x x x x
                self.pool_filter = lambda arr: arr[len(arr) : len(arr) // 2 - 1 : -1]
            elif self.pool_filter == "eo_even":
                # 0 1 2 3 4 5 6 7
                # x   x   x   x
                self.pool_filter = lambda arr: arr[0::2]
            elif self.pool_filter == "eo_odd":
                # 0 1 2 3 4 5 6 7
                #   x   x   x   x
                self.pool_filter = lambda arr: arr[1::2]
            elif self.pool_filter == "inside":
                # 0 1 2 3 4 5 6 7
                #     x x x x
                self.pool_filter = lambda arr: arr[
                    len(arr) // 2 - len(arr) // 4 : len(arr) // 2 + len(arr) // 4 : 1
                ]  # inside
            elif self.pool_filter == "outside":
                # 0 1 2 3 4 5 6 7
                # x x         x x
                self.pool_filter = lambda arr: [
                    item
                    for item in arr
                    if not (
                        item
                        in arr[
                            len(arr) // 2
                            - len(arr) // 4 : len(arr) // 2
                            + len(arr) // 4 : 1
                        ]
                    )
                ]  # outside

        graphs = {}
        layer = 1
        Qc_l = [i + 1 for i in range(n_wires)]  # We label the nodes from 1 to n
        Qp_l = Qc_l.copy()
        while len(Qc_l) > 1:

            nq_avaiable = len(Qc_l)
            mod_nq = lambda x: x % nq_avaiable
            Ec_l = [(Qc_l[i], Qc_l[mod_nq(i + self.s_c)]) for i in range(nq_avaiable)]
            if len(Ec_l) == 2 and Ec_l[0][0:] == Ec_l[1][1::-1]:
                Ec_l = [Ec_l[0]]
            measured_q = self.pool_filter(Qc_l)
            remaining_q = [q for q in Qc_l if not (q in measured_q)]
            Ep_l = [
                (measured_q[i], remaining_q[(i + self.s_p) % len(remaining_q)])
                for i in range(len(measured_q))
            ]
            # Convolution graph
            C_l = (Qc_l, Ec_l)
            # Pooling graph
            P_l = (Qp_l, Ep_l)
            # Graph for layer
            G_l = (C_l, P_l)
            graphs[layer] = G_l
            # set avaiable qubits for next layer
            layer = layer + 1
            Qc_l = [j for (i, j) in Ep_l]
            Qp_l = Qc_l.copy()
        return graphs


# Default circuit blocks
def V(symbols, bits):
    circuit = cirq.Circuit()
    circuit += cirq.rz(symbols[0]).on(bits[0]).controlled_by(bits[1])
    circuit += cirq.X(bits[0])
    circuit += cirq.rx(symbols[1]).on(bits[0]).controlled_by(bits[1])
    return circuit


def U(symbols, bits):
    circuit = cirq.Circuit()
    circuit += cirq.rx(symbols[0]).on(bits[0])
    circuit += cirq.rx(symbols[1]).on(bits[1])
    circuit += cirq.rz(symbols[2]).on(bits[0])
    circuit += cirq.rz(symbols[3]).on(bits[1])
    circuit += cirq.rz(symbols[4]).on(bits[1]).controlled_by(bits[0])
    circuit += cirq.rz(symbols[5]).on(bits[0]).controlled_by(bits[1])
    circuit += cirq.rx(symbols[6]).on(bits[0])
    circuit += cirq.rx(symbols[7]).on(bits[1])
    circuit += cirq.rz(symbols[8]).on(bits[0])
    circuit += cirq.rz(symbols[9]).on(bits[1])
    return circuit
