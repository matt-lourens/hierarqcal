from collections.abc import Sequence
import numpy as np
from qutip.qip.circuit import QubitCircuit, Gate


# TODO remove these functions, use defaults differently
# Default pooling circuit
def U(bits, symbols=None):
    circuit = QubitCircuit(len(bits))
    circuit.add_gate("RZ", controls=bits[0], targets=bits[1], control_value=symbols[0])
    # q0, q1 = cirq.LineQubit(bits[0]), cirq.LineQubit(bits[1])
    # # circuit += cirq.H(q0)
    # # circuit += cirq.H(q1)
    # circuit += cirq.rz(symbols[0]).on(q1).controlled_by(q0)
    # circuit += cirq.rz(symbols[1]).on(q0).controlled_by(q1)
    return circuit


def V(bits, symbols=None):
    circuit = QubitCircuit(len(bits))
    circuit.add_gate("CNOT", controls=bits[0], targets=bits[1])
    # circuit = cirq.Circuit()
    # q0, q1 = cirq.LineQubit(bits[0]), cirq.LineQubit(bits[1])
    # circuit += cirq.CNOT(q0, q1)
    return circuit


class Q_Primitive:
    """QCNN Primitive operation class, each instance represents a directed graph that has pointers to its predecessor and successor.
    Each directed graph corresponds to some primitive operation of a QCNN such as a convolution or pooling.
    """

    def __init__(
        self, Q=[], E=[], prev_graph=None, next_graph=None, function_mapping=None
    ):
        """Initialize primitive operation

        Args:
            Q (list(int), optional): qubits (nodes of directed graph) available for the primitive operation. Defaults to [].
            E (list(tuple(int)), optional): pairs of qubits (edges of directed graph) used for the primitive operation. Defaults to ().
            prev_graph (Q_Primitive or int, optional): Instance of previous Q_Primitive, if int the it's assumed to be the first layer and
                the int corresponds to the number of available qubits. Defaults to None.
            next_graph (Q_Primitive, optional): Instance of next Q_Primitive. Defaults to None.
            function_mapping (tuple(func,int), optional): tuple containing unitary function corresponding to primitive along with the number of paramaters
                it requieres. Defaults to None.
        """
        self.Q = Q
        self.E = E
        if isinstance(prev_graph, Q_Primitive):
            self.prev_graph = prev_graph
        else:
            # Case for first graph in sequence, then there is no previous graph and int is recieved
            self.prev_graph = None
        self.function_mapping = function_mapping
        self.next_graph = next_graph

    def set_next(self, next_graph):
        """Function to point to next primitive operation (next layer).

        Args:
            next_graph (Q_Primitive): Instance of next primitive operation
        """
        self.next_graph = next_graph


class QConv(Q_Primitive):
    def __init__(self, prev_graph, stride=1, convolution_mapping=None):

        # TODO repr functions for both conv and pool
        # TODO graphing functions for both
        if isinstance(prev_graph, Q_Primitive):
            prev_graph.set_next(self)
            Qc_l = prev_graph.Q_avail
        elif type(prev_graph) == int:
            # Assume number of qubits were specified as prev_graph for first layer
            Qc_l = [i + 1 for i in range(prev_graph)]
        elif isinstance(prev_graph, Sequence):
            # Qc_l is given as a sequence: list, tuple or range object. 
            Qc_l = list(prev_graph)
        else:
            TypeError(
                f"prev_graph needs to be int, sequence or Q_Primitive, recieved {type(prev_graph)}"
            )

        self.type = "convolution"
        self.stride = stride
        self.Q_avail = Qc_l
        # Determine convolution operation
        nq_avaiable = len(Qc_l)
        if nq_avaiable == stride:
            raise ValueError(
                f"Stride and number of avaiable qubits can't be the same, recieved:\nstride: {stride}\navaiable qubits:{nq_avaiable}"
            )
        mod_nq = lambda x: x % nq_avaiable
        Ec_l = [(Qc_l[i], Qc_l[mod_nq(i + self.stride)]) for i in range(nq_avaiable)]
        if len(Ec_l) == 2 and Ec_l[0][0:] == Ec_l[1][1::-1]:
            Ec_l = [Ec_l[0]]
        # Specify sequence of gates:
        if convolution_mapping is None:
            # default convolution layer is defined as U with 10 paramaters.
            convolution_mapping = (U, 1)
        # Initialize graph
        super().__init__(Qc_l, Ec_l, prev_graph, function_mapping=convolution_mapping)


class QPool(Q_Primitive):
    def __init__(self, prev_graph, stride=0, pool_filter="right", pooling_mapping=None):
        if isinstance(prev_graph, Q_Primitive):
            Qp_l = prev_graph.Q_avail
        elif type(prev_graph) == int:
            # Assume number of qubits were specified as prev_graph for first layer
            Qp_l = [i + 1 for i in range(prev_graph)]
        elif isinstance(prev_graph, Sequence):
            # Qc_l is given as a sequence: list, tuple or range object. 
            Qp_l = list(prev_graph)
        else:
            TypeError(
                f"prev_graph needs to be int, sequence or Q_Primitive, recieved {type(prev_graph)}"
            )
        if len(Qp_l) > 1:
            if isinstance(prev_graph, Q_Primitive):
                prev_graph.set_next(self)
            self.type = "pooling"
            self.stride = stride
            self.pool_filter_fn = self.get_pool_filter_fn(pool_filter)
            measured_q = self.pool_filter_fn(Qp_l)
            remaining_q = [q for q in Qp_l if not (q in measured_q)]
            Ep_l = [
                (measured_q[i], remaining_q[(i + self.stride) % len(remaining_q)])
                for i in range(len(measured_q))
            ]
            self.Q_avail = remaining_q
            # Specify sequence of gates:
            if pooling_mapping is None:
                pooling_mapping = (V, 0)
            # Initialize graph
            super().__init__(Qp_l, Ep_l, prev_graph, function_mapping=pooling_mapping)
        else:
            raise ValueError(
                "Pooling operation not added, Cannot perform pooling on 1 qubit"
            )

    def get_pool_filter_fn(self, pool_filter):
        if type(pool_filter) is str:
            # Mapping words to the filter type
            if pool_filter == "left":
                # 0 1 2 3 4 5 6 7
                # x x x x
                pool_filter_fn = lambda arr: arr[0 : len(arr) // 2 : 1]
            elif pool_filter == "right":
                # 0 1 2 3 4 5 6 7
                #         x x x x
                pool_filter_fn = lambda arr: arr[len(arr) : len(arr) // 2 - 1 : -1]
            elif pool_filter == "even":
                # 0 1 2 3 4 5 6 7
                # x   x   x   x
                pool_filter_fn = lambda arr: arr[0::2]
            elif pool_filter == "odd":
                # 0 1 2 3 4 5 6 7
                #   x   x   x   x
                pool_filter_fn = lambda arr: arr[1::2]
            elif pool_filter == "inside":
                # 0 1 2 3 4 5 6 7
                #     x x x x
                pool_filter_fn = (
                    lambda arr: arr[
                        len(arr) // 2
                        - len(arr) // 4 : len(arr) // 2
                        + len(arr) // 4 : 1
                    ]
                    if len(arr) > 2
                    else [arr[1]]
                )  # inside
            elif pool_filter == "outside":
                # 0 1 2 3 4 5 6 7
                # x x         x x
                pool_filter_fn = (
                    lambda arr: [
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
                    ]
                    if len(arr) > 2
                    else [arr[0]]
                )  # outside
            else:
                # Assume filter is in form contains a string specifying which indices to remove
                # For example "01001" removes idx 1 and 4 or qubit 2 and 5
                # The important thing here is for pool filter to be the same length as the current number of qubits
                # TODO add functionality to either pad or infer a filter from a string such as "101"
                pool_filter_fn = lambda arr: [
                    item
                    for item, indicator in zip(arr, pool_filter)
                    if indicator == "1"
                ]
            return pool_filter_fn


# construct reverse binary tree
def binary_tree_r(
    n_q=8,
    s_c=1,
    s_p=0,
    pool_filter="right",
    convolution_mapping=None,
    pooling_mapping=None,
):
    tail_graph = n_q
    for layer in range(1, int(np.log2(n_q)) + 1, 1):
        # Convolution
        if not (convolution_mapping is None):
            convolution_l = convolution_mapping.get(layer, convolution_mapping[1])
        else:
            convolution_l = None
        tail_graph = QConv(tail_graph, stride=s_c, convolution_mapping=convolution_l)
        if tail_graph.prev_graph is None:
            # Set first graph, i.e. first layer first convolution
            head_graph = tail_graph
        # Pooling
        if not (pooling_mapping is None):
            pooling_l = pooling_mapping.get(layer, pooling_mapping[1])
        else:
            pooling_l = None
        tail_graph = QPool(
            tail_graph,
            stride=s_p,
            pool_filter=pool_filter,
            pooling_mapping=pooling_l,
        )
    return head_graph, tail_graph
