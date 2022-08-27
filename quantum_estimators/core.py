import numpy as np
import cirq

# TODO remove these functions, use defaults differently
# Default pooling circuit
def U(bits, symbols=None):
    circuit = cirq.Circuit()
    q0, q1 = cirq.LineQubit(bits[0]), cirq.LineQubit(bits[1])
    # circuit += cirq.H(q0)
    # circuit += cirq.H(q1)
    circuit += cirq.rz(symbols[0]).on(q1).controlled_by(q0)
    # circuit += cirq.rz(symbols[1]).on(q0).controlled_by(q1)
    return circuit


def V(bits, symbols=None):
    circuit = cirq.Circuit()
    q0, q1 = cirq.LineQubit(bits[0]), cirq.LineQubit(bits[1])
    circuit += cirq.CNOT(q0, q1)
    return circuit


class DiGraph:
    def __init__(
        self, Q=(), E=(), prev_graph=None, next_graph=None, function_mapping=None
    ):
        # TODO should this rather be a namedtuple?
        self.Q = Q
        self.E = E
        if isinstance(prev_graph, DiGraph):
            self.prev_graph = prev_graph
        else:
            self.prev_graph = None
        self.function_mapping = function_mapping
        self.next_graph = next_graph

    def set_next(self, next_graph):
        self.next_graph = next_graph


class QConv(DiGraph):
    def __init__(self, prev_graph, stride=1, convolution_mapping=None):
        # TODO test that stride < Q
        # TODO test that stride is odd (this can be a more involved test if it's not even numbered qubits)
        # TODO repr functions for both conv and pool
        # TODO graphing functions for both
        if isinstance(prev_graph, DiGraph):
            prev_graph.set_next(self)
        self.type = "convolution"
        self.stride = stride
        if type(prev_graph) == int:
            # assume first layer and number of qubits were specified
            Q = [i + 1 for i in range(prev_graph)]
        else:
            # Sorting is important because of the way pooling filters gets implemented
            if prev_graph.type == "pooling":
                # All nodes that were not measured, the i indices are the measured qubits for a pooling layer
                Q = sorted(
                    list(set(prev_graph.Q) - set([i for (i, j) in prev_graph.E]))
                )
            else:
                Q = sorted(prev_graph.Q)
        # Determine convolution operation
        nq_avaiable = len(Q)
        mod_nq = lambda x: x % nq_avaiable
        Ec_l = [(Q[i], Q[mod_nq(i + self.stride)]) for i in range(nq_avaiable)]
        if len(Ec_l) == 2 and Ec_l[0][0:] == Ec_l[1][1::-1]:
            Ec_l = [Ec_l[0]]
        # Specify sequence of gates:
        if convolution_mapping is None:
            # default convolution layer is defined as U with 10 paramaters.
            convolution_mapping = (U, 1)
        # Initialize graph
        super().__init__(Q, Ec_l, prev_graph, function_mapping=convolution_mapping)


class QPool(DiGraph):
    def __init__(self, prev_graph, stride=0, pool_filter="right", pooling_mapping=None):
        if isinstance(prev_graph, DiGraph):
            prev_graph.set_next(self)
        self.type = "pooling"
        self.stride = stride
        self.pool_filter_fn = self.get_pool_filter_fn(pool_filter)
        # Determine pooling operation
        if prev_graph.type == "pooling":
            # All nodes that were not measured, the i indices are the measured qubits for a pooling layer
            Qp_l = sorted(list(set(prev_graph.Q) - set([i for (i, j) in prev_graph.E])))
        else:
            Qp_l = sorted(prev_graph.Q)
        # Qc_l = prev_graph.Q
        # Qp_l = Qc_l.copy()
        measured_q = self.pool_filter_fn(Qp_l)
        remaining_q = [q for q in Qp_l if not (q in measured_q)]
        Ep_l = [
            (measured_q[i], remaining_q[(i + self.stride) % len(remaining_q)])
            for i in range(len(measured_q))
        ]
        # Specify sequence of gates:
        if pooling_mapping is None:
            pooling_mapping = (V, 0)
        # Initialize graph
        super().__init__(Qp_l, Ep_l, prev_graph, function_mapping=pooling_mapping)

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
