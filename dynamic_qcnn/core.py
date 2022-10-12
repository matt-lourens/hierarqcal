from collections.abc import Sequence
from enum import Enum
import warnings
from copy import copy, deepcopy
from collections import deque
import numpy as np
import itertools as it
from qutip.qip.circuit import QubitCircuit, Gate
import cirq


# TODO remove these functions, use defaults differently
# Default pooling circuit
def U(bits, symbols=None):
    # circuit = QubitCircuit(len(bits))
    # circuit.add_gate("RZ", controls=bits[0], targets=bits[1], control_value=symbols[0])
    q0, q1 = cirq.LineQubit(bits[0]), cirq.LineQubit(bits[1])
    circuit = cirq.Circuit()
    # circuit += cirq.H(q0)
    # circuit += cirq.H(q1)
    circuit += cirq.rz(symbols[0]).on(q1).controlled_by(q0)
    # circuit += cirq.rz(symbols[1]).on(q0).controlled_by(q1)
    return circuit


def V(bits, symbols=None):
    # circuit = QubitCircuit(len(bits))
    # circuit.add_gate("CNOT", controls=bits[0], targets=bits[1])
    circuit = cirq.Circuit()
    q0, q1 = cirq.LineQubit(bits[0]), cirq.LineQubit(bits[1])
    circuit += cirq.CNOT(q0, q1)
    return circuit


class Primitive_Types(Enum):
    CONVOLUTION = "convolution"
    POOLING = "pooling"
    DENSE = "dense"


class Qmotif:
    def __init__(
        self,
        Q=[],
        E=[],
        Q_avail=[],
        next=None,
        prev=None,
        function_mapping=None,
        is_default_mapping=True,
        is_operation=True,
    ) -> None:
        # Meta information
        self.is_operation = is_operation
        self.is_default_mapping = is_default_mapping
        # Data capturing
        self.Q = Q
        self.Q_avail = Q_avail
        self.E = E
        self.function_mapping = function_mapping
        # pointers
        self.prev = prev
        self.next = next

    def __add__(self, other):
        return self.append(other)

    def __mul__(self, other):
        # TODO must create new each time investigate __new__, this copies the object.
        return Qmotifs((deepcopy(self) for i in range(other)))

    def append(self, other):
        return Qmotifs((deepcopy(self), deepcopy(other)))

    def set_Q(self, Q):
        self.Q = Q

    def set_E(self, E):
        self.E = E

    def set_Qavail(self, Q_avail):
        self.Q_avail = Q_avail

    def set_mapping(self, function_mapping):
        self.function_mapping = function_mapping

    def set_next(self, next):
        self.next = next

    def set_prev(self, prev):
        self.prev = prev


class Qmotifs(tuple):
    # TODO mention assumption that only operators should be used i.e. +, *
    # TODO explain this hackery, it's to ensure the case (a,b)+b -> (a,b,c) no matter type of b
    def __add__(self, other):
        if isinstance(other, Sequence):
            return Qmotifs(tuple(self) + tuple(other))
        else:
            return Qmotifs(tuple(self) + (other,))

    def __mul__(self, other):
        # repeats "other=int" times i.e. other=5 -> i in range(5)
        if type(other) is int:
            return Qmotifs((deepcopy(item) for i in range(other) for item in self))
        else:
            raise ValueError("Only integers are allowed for multiplication")


class Qconv(Qmotif):
    def __init__(self, stride=1, step=1, offset=0, convolution_mapping=None):
        self.type = Primitive_Types.CONVOLUTION.value
        self.stride = stride
        self.step = step
        self.offset = offset
        # Specify sequence of gates:
        if convolution_mapping is None:
            # default convolution layer is defined as U with 1 paramater.
            convolution_mapping = (U, 1)
            is_default_mapping = True
        else:
            is_default_mapping = False
        # Initialize graph
        super().__init__(
            function_mapping=convolution_mapping, is_default_mapping=is_default_mapping
        )

    def __call__(self, Qc_l, *args, **kwds):
        # Determine convolution operation
        nq_avaiable = len(Qc_l)
        if self.stride % nq_avaiable == 0:
            # TODO make this clear in documentation
            # warnings.warn(
            #     f"Stride and number of avaiable qubits can't be the same, recieved:\nstride: {self.stride}\navaiable qubits:{nq_avaiable}. Deafulting to stride of 1"
            # )
            self.stride = 1
        mod_nq = lambda x: x % nq_avaiable
        Ec_l = [
            (Qc_l[mod_nq(i)], Qc_l[mod_nq(i + self.stride)])
            for i in range(self.offset, nq_avaiable, self.step)
        ]
        if len(Ec_l) == 2 and Ec_l[0][0:] == Ec_l[1][1::-1]:
            Ec_l = [Ec_l[0]]
        self.set_Q(Qc_l)
        self.set_E(Ec_l)
        # All qubits are still available for the next operation
        self.set_Qavail(Qc_l)
        mapping = kwds.get("mapping", None)
        if mapping:
            self.set_mapping(mapping)
        return self


class Qdense(Qmotif):
    """Dense layer, connects unitaries to all possible combinations of wires"""

    def __init__(self, permutations=False, function_mapping=None):
        self.type = Primitive_Types.DENSE.value
        self.permutations = permutations
        # Specify sequence of gates:
        if function_mapping is None:
            # default convolution layer is defined as U with 10 paramaters.
            function_mapping = (U, 1)
            is_default_mapping = True
        else:
            is_default_mapping = False
        # Initialize graph
        super().__init__(
            function_mapping=function_mapping, is_default_mapping=is_default_mapping
        )

    def __call__(self, Qc_l, *args, **kwds):
        # All possible wire combinations
        if self.permutations:
            Ec_l = list(it.permutations(Qc_l, r=2))
        else:
            Ec_l = list(it.combinations(Qc_l, r=2))
        if len(Ec_l) == 2 and Ec_l[0][0:] == Ec_l[1][1::-1]:
            Ec_l = [Ec_l[0]]
        self.set_Q(Qc_l)
        self.set_E(Ec_l)
        # All qubits are still available for the next operation
        self.set_Qavail(Qc_l)
        mapping = kwds.get("mapping", None)
        if mapping:
            self.set_mapping(mapping)
        return self


class Qpool(Qmotif):
    def __init__(self, stride=0, filter="right", pooling_mapping=None):
        self.type = Primitive_Types.POOLING.value
        self.stride = stride
        self.pool_filter_fn = self.get_pool_filter_fn(filter)
        # Specify sequence of gates:
        if pooling_mapping is None:
            pooling_mapping = (V, 0)
            is_default_mapping = True
        else:
            is_default_mapping = False
        # Initialize graph
        super().__init__(
            function_mapping=pooling_mapping, is_default_mapping=is_default_mapping
        )

    def __call__(self, Qp_l, *args, **kwds):
        if len(Qp_l) > 1:
            measured_q = self.pool_filter_fn(Qp_l)
            remaining_q = [q for q in Qp_l if not (q in measured_q)]
            if len(remaining_q) > 0:
                Ep_l = [
                    (measured_q[i], remaining_q[(i + self.stride) % len(remaining_q)])
                    for i in range(len(measured_q))
                ]
            else:
                # No qubits were pooled
                Ep_l = []
                remaining_q = Qp_l

        else:
            # raise ValueError(
            #     "Pooling operation not added, Cannot perform pooling on 1 qubit"
            # )
            # No qubits were pooled
            # TODO make clear in documentation, no pooling is done if 1 qubit remain
            Ep_l = []
            remaining_q = Qp_l
        self.set_Q(Qp_l)
        self.set_E(Ep_l)
        self.set_Qavail(remaining_q)
        mapping = kwds.get("mapping", None)
        if mapping:
            self.set_mapping(mapping)
        return self

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
        else:
            pool_filter_fn = pool_filter
        return pool_filter_fn


class Qcnn:
    def __init__(self, qubits, function_mappings={}) -> None:
        # Set available qubit
        if isinstance(qubits, Qmotif):
            self.tail = qubits
            self.head = self.tail
        else:
            self.tail = Qfree(qubits)
            self.head = self.tail
        self.function_mappings = function_mappings
        self.mapping_counter = {
            primitive_type.value: 1 for primitive_type in Primitive_Types
        }

    def append(self, motif):
        motif = deepcopy(motif)

        if motif.is_operation & motif.is_default_mapping:
            mapping = None
            # If no function mapping was provided
            mappings = self.function_mappings.get(motif.type, None)
            if mappings:
                mapping = mappings[
                    (self.mapping_counter.get(motif.type) - 1) % len(mappings)
                ]
                self.mapping_counter.update(
                    {motif.type: self.mapping_counter.get(motif.type) + 1}
                )
            motif(self.head.Q_avail, mapping=mapping)
        else:
            motif(self.head.Q_avail)
        new_qcnn = deepcopy(self)
        new_qcnn.head.set_next(motif)
        new_qcnn.head = new_qcnn.head.next
        return new_qcnn

    def extend(self, motifs):
        new_qcnn = deepcopy(self)
        for motif in motifs:
            new_qcnn = new_qcnn.append(motif)
        return new_qcnn

    def merge(self, qcnn):
        # ensure immutability
        other_qcnn = deepcopy(qcnn)
        new_qcnn = deepcopy(self)
        other_qcnn.update_Q(new_qcnn.head.Q_avail)
        new_qcnn.head.set_next(other_qcnn.tail)
        new_qcnn.head = other_qcnn.head
        return new_qcnn

    def extmerge(self, qcnns):
        new_qcnn = deepcopy(self)
        for qcnn in qcnns:
            new_qcnn = new_qcnn.merge(qcnn)
        return new_qcnn

    def update_Q(self, Q):
        motif = self.tail(Q)
        while motif.next is not None:
            motif = motif.next(motif.Q_avail)

    def __add__(self, other):
        if isinstance(other, Qcnn):
            new_qcnn = self.merge(other)
        elif isinstance(other, Sequence):
            if isinstance(other[-1], Qmotif):
                new_qcnn = self.extend(other)
            elif isinstance(other[-1], Qcnn):
                new_qcnn = self.extmerge(other)
        elif isinstance(other, Qmotif):
            new_qcnn = self.append(other)
        return new_qcnn

    def __mul__(self, other):
        # TODO
        if isinstance(other, Qcnn):
            new_qcnn = self.merge(other)
        elif isinstance(other, Sequence):
            new_qcnn = self.extend(other)
        elif isinstance(other, Qmotif):
            new_qcnn = self.append(other)
        elif isinstance(other, int):
            t = (self,) * other
            if other > 1:
                new_qcnn = t[0] + t[1:]
            else:
                # TODO no action for multiplication too large, might want to make use of 0
                new_qcnn = self
        return new_qcnn

    def __iter__(self):
        # Generator to go from head to tail and only return operations
        current = self.tail
        while current is not None:
            if current.is_operation:
                yield current
            current = current.next


class Qfree(Qmotif):
    """Frees up a number of Qbits

    Args:
        Qmotif (_type_): _description_
    """

    def __init__(self, Q) -> None:
        if isinstance(Q, Sequence):
            Qfree = Q
        elif type(Q) == int:
            Qfree = [i + 1 for i in range(Q)]
        self.type = "special"
        # Initialize graph
        super().__init__(Q=Qfree, Q_avail=Qfree, is_operation=False)

    def __add__(self, other):
        return Qcnn(self) + other

    def __call__(self, Q):
        # TODO doesn't do anything If a motif needs updating based on merge
        self.set_Q(self.Q)
        self.set_Qavail(self.Q)
        return self


class LinkedDiGraph:
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
        # Data
        self.Q = Q
        self.E = E
        self.function_mapping = function_mapping
        if isinstance(prev_graph, LinkedDiGraph):
            self.prev_graph = prev_graph
        else:
            # Case for first graph in sequence, then there is no previous graph and int is recieved
            self.prev_graph = None
            self.tail = self
        if isinstance(next_graph, LinkedDiGraph):
            self.next_graph = next_graph
        else:
            # Case for first graph in sequence, then there is no previous graph and int is recieved
            self.next_graph = None
            self.tail = self

        self.next_graph = next_graph

    def set_next(self, next_graph):
        """Function to point to next primitive operation (next layer).

        Args:
            next_graph (Q_Primitive): Instance of next primitive operation
        """
        self.next_graph = next_graph

    def __call__(self, prev_graph):
        self.prev_graph = prev_graph


# print("debug")
# # Pooling circuit
# def V_1(bits, symbols=None):  # 1
#     circuit = cirq.Circuit()
#     q0, q1 = cirq.LineQubit(bits[0]), cirq.LineQubit(bits[1])
#     circuit += cirq.rx(symbols[0]).on(q1).controlled_by(q0)

#     return circuit


# def V_2(bits, symbols=None):  # 0
#     circuit = cirq.Circuit()
#     q0, q1 = cirq.LineQubit(bits[0]), cirq.LineQubit(bits[1])
#     circuit += cirq.CNOT(q0, q1)
#     return circuit


# # Convolution circuit
# def U_1(bits, symbols=None):  # 1
#     circuit = cirq.Circuit()
#     q0, q1 = cirq.LineQubit(bits[0]), cirq.LineQubit(bits[1])
#     circuit += cirq.rz(symbols[0]).on(q1).controlled_by(q0)
#     return circuit


# def U_2(bits, symbols=None):  # 1
#     circuit = cirq.Circuit()
#     q0, q1 = cirq.LineQubit(bits[0]), cirq.LineQubit(bits[1])
#     circuit += cirq.ry(symbols[0]).on(q1).controlled_by(q0)
#     return circuit


# function_mappings = {
#     "convolution": [(U_1, 1), (U_2, 1)],
#     "pooling": [(V_1, 1), (V_2, 0)],
# }
# qcnn = Qcnn(16, function_mappings=function_mappings)

# a = qcnn + (Qconv() + Qpool()) * 3

# import random
# import operator
# from functools import reduce
# from dynamic_qcnn import convert_graph_to_circuit_cirq

# N = 8

# p = [
#     Qpool(s_p, filter=i_filter)
#     for i_filter in (
#         "right",
#         "left",
#         "even",
#         "odd",
#         "inside",
#         "outside",
#         lambda arr: [arr[-1]],
#         lambda arr: [arr[0]],
#     )
#     for s_p in range(N)
# ]
# c = [Qconv(s_c, s_t,so) for s_c in range(N) for s_t in range(1, N, 1) for so in range(N)]
# m2_1 = [random.choice(c) + random.choice(p) for i in range(int(np.log2(N)))]
# m2_2 = [random.choice(c) + random.choice(c) for i in range(int(np.log2(N)))]
# m3_1 = reduce(operator.add, m2_1)
# Qfree(N) + m3_1
# circuit, symbols = convert_graph_to_circuit_cirq(Qfree(N) + m3_1)
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
        tail_graph = Qconv(tail_graph, stride=s_c, convolution_mapping=convolution_l)
        if tail_graph.prev_graph is None:
            # Set first graph, i.e. first layer first convolution
            head_graph = tail_graph
        # Pooling
        if not (pooling_mapping is None):
            pooling_l = pooling_mapping.get(layer, pooling_mapping[1])
        else:
            pooling_l = None
        tail_graph = Qpool(
            tail_graph,
            stride=s_p,
            pool_filter=pool_filter,
            pooling_mapping=pooling_l,
        )
    return head_graph, tail_graph


# class QConv(QMotif):
#     def __init__(self, prev_graph, stride=1, convolution_mapping=None):

#         # TODO repr functions for both conv and pool
#         # TODO graphing functions for both
#         if isinstance(prev_graph, Sequence):
#             # Qc_l is given as a sequence: list, tuple or range object.
#             if isinstance(prev_graph[-1], Q_Primitive):
#                 prev_graph[-1].set_next(self)
#                 Qc_l = prev_graph[-1].Q_avail
#             elif type(prev_graph[-1]) == int:
#                 # Assume number of qubits were specified as prev_graph for first layer
#                 Qc_l = list(prev_graph)
#         else:
#             if isinstance(prev_graph, Q_Primitive):
#                 prev_graph.set_next(self)
#                 Qc_l = prev_graph.Q_avail
#             elif type(prev_graph) == int:
#                 # Assume number of qubits were specified as prev_graph for first layer
#                 Qc_l = [i + 1 for i in range(prev_graph)]
#         # elif isinstance(prev_graph, Sequence):
#         #     # Qc_l is given as a sequence: list, tuple or range object.

#         #     Qc_l = list(prev_graph)
#         # else:
#         #     TypeError(
#         #         f"prev_graph needs to be int, sequence or Q_Primitive, recieved {type(prev_graph)}"
#         #     )

#         self.type = "convolution"
#         self.stride = stride
#         self.Q_avail = Qc_l
#         # Determine convolution operation
#         nq_avaiable = len(Qc_l)
#         if nq_avaiable == stride:
#             raise ValueError(
#                 f"Stride and number of avaiable qubits can't be the same, recieved:\nstride: {stride}\navaiable qubits:{nq_avaiable}"
#             )
#         mod_nq = lambda x: x % nq_avaiable
#         Ec_l = [(Qc_l[i], Qc_l[mod_nq(i + self.stride)]) for i in range(nq_avaiable)]
#         if len(Ec_l) == 2 and Ec_l[0][0:] == Ec_l[1][1::-1]:
#             Ec_l = [Ec_l[0]]
#         # Specify sequence of gates:
#         if convolution_mapping is None:
#             # default convolution layer is defined as U with 10 paramaters.
#             convolution_mapping = (U, 1)
#         # Initialize graph
#         super().__init__(Qc_l, Ec_l, prev_graph, function_mapping=convolution_mapping)

# class QPool(Q_Primitive):
#     def __init__(self, prev_graph, stride=0, pool_filter="right", pooling_mapping=None):
#         if isinstance(prev_graph, Q_Primitive):
#             Qp_l = prev_graph.Q_avail
#         elif type(prev_graph) == int:
#             # Assume number of qubits were specified as prev_graph for first layer
#             Qp_l = [i + 1 for i in range(prev_graph)]
#         elif isinstance(prev_graph, Sequence):
#             # Qc_l is given as a sequence: list, tuple or range object.
#             Qp_l = list(prev_graph)
#         else:
#             TypeError(
#                 f"prev_graph needs to be int, sequence or Q_Primitive, recieved {type(prev_graph)}"
#             )
#         if len(Qp_l) > 1:
#             if isinstance(prev_graph, Q_Primitive):
#                 prev_graph.set_next(self)
#             self.type = "pooling"
#             self.stride = stride
#             self.pool_filter_fn = self.get_pool_filter_fn(pool_filter)
#             measured_q = self.pool_filter_fn(Qp_l)
#             remaining_q = [q for q in Qp_l if not (q in measured_q)]
#             Ep_l = [
#                 (measured_q[i], remaining_q[(i + self.stride) % len(remaining_q)])
#                 for i in range(len(measured_q))
#             ]
#             self.Q_avail = remaining_q
#             # Specify sequence of gates:
#             if pooling_mapping is None:
#                 pooling_mapping = (V, 0)
#             # Initialize graph
#             super().__init__(Qp_l, Ep_l, prev_graph, function_mapping=pooling_mapping)
#         else:
#             raise ValueError(
#                 "Pooling operation not added, Cannot perform pooling on 1 qubit"
#             )

#     def get_pool_filter_fn(self, pool_filter):
#         if type(pool_filter) is str:
#             # Mapping words to the filter type
#             if pool_filter == "left":
#                 # 0 1 2 3 4 5 6 7
#                 # x x x x
#                 pool_filter_fn = lambda arr: arr[0 : len(arr) // 2 : 1]
#             elif pool_filter == "right":
#                 # 0 1 2 3 4 5 6 7
#                 #         x x x x
#                 pool_filter_fn = lambda arr: arr[len(arr) : len(arr) // 2 - 1 : -1]
#             elif pool_filter == "even":
#                 # 0 1 2 3 4 5 6 7
#                 # x   x   x   x
#                 pool_filter_fn = lambda arr: arr[0::2]
#             elif pool_filter == "odd":
#                 # 0 1 2 3 4 5 6 7
#                 #   x   x   x   x
#                 pool_filter_fn = lambda arr: arr[1::2]
#             elif pool_filter == "inside":
#                 # 0 1 2 3 4 5 6 7
#                 #     x x x x
#                 pool_filter_fn = (
#                     lambda arr: arr[
#                         len(arr) // 2
#                         - len(arr) // 4 : len(arr) // 2
#                         + len(arr) // 4 : 1
#                     ]
#                     if len(arr) > 2
#                     else [arr[1]]
#                 )  # inside
#             elif pool_filter == "outside":
#                 # 0 1 2 3 4 5 6 7
#                 # x x         x x
#                 pool_filter_fn = (
#                     lambda arr: [
#                         item
#                         for item in arr
#                         if not (
#                             item
#                             in arr[
#                                 len(arr) // 2
#                                 - len(arr) // 4 : len(arr) // 2
#                                 + len(arr) // 4 : 1
#                             ]
#                         )
#                     ]
#                     if len(arr) > 2
#                     else [arr[0]]
#                 )  # outside
#             else:
#                 # Assume filter is in form contains a string specifying which indices to remove
#                 # For example "01001" removes idx 1 and 4 or qubit 2 and 5
#                 # The important thing here is for pool filter to be the same length as the current number of qubits
#                 # TODO add functionality to either pad or infer a filter from a string such as "101"
#                 pool_filter_fn = lambda arr: [
#                     item
#                     for item, indicator in zip(arr, pool_filter)
#                     if indicator == "1"
#                 ]
#             return pool_filter_fn
