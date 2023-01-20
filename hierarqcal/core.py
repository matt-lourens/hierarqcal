"""
This module contains the core classes for the hierarqcal package. Qmotif is the base class for all motifs, Qmotifs is a sequence of motifs and Qcnn is the full quantum circuit architecture, that handles the interaction between motifs.

Create a qcnn as follows:

.. code-block:: python

    from hierarqcal import Qfree, Qconv, Qpool
    my_qcnn = Qfree(8) + (Qconv(1) + Qpool(filter="right")) * 3

The above creates a qcnn that resembles a reverse binary tree architecture. There are 8 free qubits and then a convolution-pooling unit is repeated three times.
"""

from collections.abc import Sequence
from enum import Enum
import warnings
from copy import copy, deepcopy
from collections import deque
import numpy as np
import itertools as it


class Primitive_Types(Enum):
    """
    Enum for primitive types
    """

    CONVOLUTION = "convolution"
    POOLING = "pooling"
    DENSE = "dense"


class Qmotif:
    """
    Quantum Circuit architectures are created by stacking motifs hierarchacially, the lowest level motifs (primitives) are building blocks for higher level ones.
    Examples of primitives are convolution (:py:class:`Qconv`), pooling (:py:class:`Qpool`) and dense (:py:class:`Qdense`) operations which inherits this class. Each motif is a directed graph with nodes
    Q for qubits and edges E for unitary operations applied between them, the direction of an edge being the order of interaction for the unitary. Each instance has
    pointers to its predecessor and successor.This class is for a single motif and the Qmotifs class is for a sequence of motifs stored as tuples, then sequences of
    motifs are again stored as tuples  This is to allow hiearchical stacking which in the end is one tuple of motifs.
    """

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
        """
        Add two motifs together, this  appends them next to each other in a tuple.

        Args:
            other (Qmotif): Motif to append to current motif (self).

        Returns:
            Qmotifs(tuple): A 2-tuple of motifs: (self, other) in the order they were added. The tuples contain copies of the original motifs.
        """
        return self.append(other)

    def __mul__(self, other):
        """
        Repeat motif "other" times.

        Args:
            other (int): Number of times to repeat motif.

        Returns:
            Qmotifs(tuple): A tuple of motifs: (self, self, ..., self), where each is a new object copied from the original.
        """
        # TODO must create new each time investigate __new__, this copies the object.
        return Qmotifs((deepcopy(self) for i in range(other)))

    def append(self, other):
        """
        Append motif to current motif.

        Args:
            other (Qmotif): Motif to append to current motif (self).

        Returns:
            Qmotifs(tuple): A 2-tuple of motifs: (self, other) in the order they were added. The tuples contain copies of the original motifs.
        """
        return Qmotifs((deepcopy(self), deepcopy(other)))

    def set_Q(self, Q):
        """
        Set the qubit labels Q of the motif.

        Args:
            Q (list(int or string)): List of qubit labels.
        """
        self.Q = Q

    def set_E(self, E):
        """
        Set the edges E of the motif.

        Args:
            E (list(tuples)): List of edges.
        """
        self.E = E

    def set_Qavail(self, Q_avail):
        """
        Set the available qubits Q_avail of the motif. This is usually caluclated by the previous motif in the stack, for example pooling removes qubits from being available and
        would use this fcuntion to update the available qubits after it's action.

        Args:
            Q_avail (list): List of available qubits.
        """
        self.Q_avail = Q_avail

    def set_mapping(self, function_mapping):
        """
        Specify the unitary operations applied according to the type of motif.

        Args:
            function_mapping (tuple(function, int)): Function mapping is specified as a tuple, where the first argument is a function and the second is the number of symbols it uses. A symbol here refers to an variational paramater for a quantum circuit, i.e. crz(theta, q0, q1) <- theta is a symbol for the gate.
        """
        self.function_mapping = function_mapping

    def set_next(self, next):
        """
        Set the next motif in the stack.

        Args:
            next (Qmotif): Next motif in the stack.
        """
        self.next = next

    def set_prev(self, prev):
        """
        Set the previous motif in the stack.

        Args:
            prev (Qmotif): Previous motif in the stack.
        """
        self.prev = prev


class Qmotifs(tuple):
    """
    A tuple of motifs, this is the data structure for storing sequences motifs. It subclasses tuple, so all tuple methods are available.
    """

    # TODO mention assumption that only operators should be used i.e. +, *
    # TODO explain this hackery, it's to ensure the case (a,b)+b -> (a,b,c) no matter type of b
    def __add__(self, other):
        """
        Add two tuples of motifs together.

        Args:
            other (Qmotifs or Qmotif): Multiple motifs or singe motif to add to current sequence of motifs.

        Returns:
            Qmotifs(tuple): A single tuple of the motifs that were added, These are copies of the original motifs, since tuples are immutable.
        """
        if isinstance(other, Sequence):
            return Qmotifs(tuple(self) + tuple(other))
        else:
            return Qmotifs(tuple(self) + (other,))

    def __mul__(self, other):
        """
        Repeat motifs "other" times.

        Args:
            other (int): Number of times to repeat motifs.

        Returns:
            Qmotifs(tuple): A tuple of motifs: (self, self, ..., self), where each is a new object copied from the original.

        Raises:
            ValueError: Only integers are allowed for multiplication.
        """
        # repeats "other=int" times i.e. other=5 -> i in range(5)
        if type(other) is int:
            return Qmotifs((deepcopy(item) for i in range(other) for item in self))
        else:
            raise ValueError("Only integers are allowed for multiplication")


class Qconv(Qmotif):
    """
    A convolution motif, used to specify convolution operations in the quantum neural network.
    """

    def __init__(self, stride=1, step=1, offset=0, convolution_mapping=None):
        self.type = Primitive_Types.CONVOLUTION.value
        self.stride = stride
        self.step = step
        self.offset = offset
        # Specify sequence of gates:
        if convolution_mapping is None:
            # default convolution layer is defined as U with 1 paramater.
            is_default_mapping = True
        else:
            is_default_mapping = False
        # Initialize graph
        super().__init__(
            function_mapping=convolution_mapping, is_default_mapping=is_default_mapping
        )

    def __call__(self, Qc_l, *args, **kwds):
        """
        Call the motif, this generates the edges and qubits of the motif (directed graph) based on it's available qubits.
        Each time a motif in the stack changes, a loop runs through the stack from the beginning and calls each motif to update the graph (the available qubits, the edges etc).

        Args:
            Qc_l (list): List of available qubits.
            *args: Variable length argument list.
            **kwds: Arbitrary keyword arguments, such as:

                * mapping (tuple(function, int)):
                    Function mapping is specified as a tuple, where the first argument is a function and the second is the number of symbols it uses. A symbol here refers to an variational paramater for a quantum circuit, i.e. crz(theta, q0, q1) <- theta is a symbol for the gate.

        Returns:
            Qconv: Returns the updated version of itself, with correct nodes and edges.
        """
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
    """
    A dense motif, it connects unitaries to all possible combinations of qubits (all possible edges given Q) in the quantum circuit.
    """

    def __init__(self, permutations=False, function_mapping=None):
        self.type = Primitive_Types.DENSE.value
        self.permutations = permutations
        # Specify sequence of gates:
        if function_mapping is None:
            # default convolution layer is defined as U with 10 paramaters.
            # function_mapping = (U, 1) TODO remove
            is_default_mapping = True
        else:
            is_default_mapping = False
        # Initialize graph
        super().__init__(
            function_mapping=function_mapping, is_default_mapping=is_default_mapping
        )

    def __call__(self, Qc_l, *args, **kwds):
        """
        Call the motif, this is used to generate the edges and qubits of the motif (directed graph) based on it's available qubits.
        Each time a motif in the stack changes, a loop runs through the stack from the beginning and calls each motif to update the graph (the available qubits, the edges etc).

        Args:
            Qc_l (list): List of available qubits.
            *args: Variable length argument list.
            **kwds: Arbitrary keyword arguments, such as:

                * mapping (tuple(function, int)):
                    Function mapping is specified as a tuple, where the first argument is a function and the second is the number of symbols it uses. A symbol here refers to an variational paramater for a quantum circuit, i.e. crz(theta, q0, q1) <- theta is a symbol for the gate.

        Returns:
            Qdense: Returns the updated version of itself, with correct nodes and edges.
        """
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
    """
    A pooling motif, it pools qubits together based (some controlled operation where the control is not used for the rest of the circuit).
    This motif changes the available qubits for the next motif in the stack.
    """

    def __init__(self, stride=0, filter="right", pooling_mapping=None):
        self.type = Primitive_Types.POOLING.value
        self.stride = stride
        self.pool_filter_fn = self.get_pool_filter_fn(filter)
        # Specify sequence of gates:
        if pooling_mapping is None:
            # pooling_mapping = (V, 0) TODO remove this line
            is_default_mapping = True
        else:
            is_default_mapping = False
        # Initialize graph
        super().__init__(
            function_mapping=pooling_mapping, is_default_mapping=is_default_mapping
        )

    def __call__(self, Qp_l, *args, **kwds):
        """
        Call the motif, this is used to generate the edges and qubits of the motif (directed graph) based on it's available qubits.
        Each time a motif in the stack changes, a loop runs through the stack from the beginning and calls each motif to update the graph (the available qubits, the edges etc).

        Args:
            Qp_l (list): List of available qubits.
            *args: Variable length argument list.
            **kwds: Arbitrary keyword arguments, such as:

                * mapping (tuple(function, int)):
                    Function mapping is specified as a tuple, where the first argument is a function and the second is the number of symbols it uses. A symbol here refers to an variational paramater for a quantum circuit, i.e. crz(theta, q0, q1) <- theta is a symbol for the gate.

        Returns:
            Qpool: Returns the updated version of itself, with correct nodes and edges.
        """
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
        """
        Get the filter function for the pooling operation.

        Args:
            pool_filter (str or lambda): The filter type, can be "left", "right", "even", "odd", "inside" or "outside" which corresponds to a specific pattern (see comments in code below).
                                            The string can also be a bitstring, i.e. "01000" which pools the 2nd qubit.
                                            If a lambda function is passed, it is used as the filter function, it should work as follow: pool_filter_fn([0,1,2,3,4,5,6,7]) -> [0,1,2,3], i.e.
                                            the function returns a sublist of the input list based on some pattern. What's nice about passing a function is that it can be list length independent,
                                            meaning the same kind of pattern will be applied as the list grows or shrinks.
        """
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
    """
    The main class that manages the "stack" of motifs, it handles the interaction between successive motifs and when a motifs are added to the stack,
    it updates all the others accordingly. An object of this class fully captures the architectural information of a Quantum convolutional neural network.
    It also handles function (unitary operation) mappings.
    """

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
        """
        Add a motif to the stack of motifs and update it (call to generate nodes and edges) according to the action of the previous motifs in the stack.

        Args:
            motif (Qmotif): The motif to add to the stack.

        Returns:
            Qcnn: A new Qcnn object with the new motif added to the stack.
        """
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
        """
        Add and update a list of motifs to the current stack of motifs call each to generate their nodes and edges according to the action of the previous motifs in the stack.

        Args:
            motifs (Qmotifs): A tuple of motifs to add to the stack.

        Returns:
            Qcnn: A new Qcnn object with the motifs added to the stack.
        """
        new_qcnn = deepcopy(self)
        for motif in motifs:
            new_qcnn = new_qcnn.append(motif)
        return new_qcnn

    def merge(self, qcnn):
        """
        Merge two Qcnn objects by adding the head of the second one to the tail of the first one. Both are copied to ensure immutability.

        Args:
            qcnn (Qcnn): The Qcnn object to merge with the current one.

        Returns:
            Qcnn: A new Qcnn object with the two merged.
        """
        # ensure immutability
        other_qcnn = deepcopy(qcnn)
        new_qcnn = deepcopy(self)
        other_qcnn.update_Q(new_qcnn.head.Q_avail)
        new_qcnn.head.set_next(other_qcnn.tail)
        new_qcnn.head = other_qcnn.head
        return new_qcnn

    def extmerge(self, qcnns):
        """
        Merge a list of Qcnn objects by adding the head of each one to the tail of the previous one. All are copied to ensure immutability.

        Args:
            qcnns (Qcnn): A list of Qcnn objects to merge with the current one.

        Returns:
            Qcnn: A new Qcnn object with the list merged.
        """
        new_qcnn = deepcopy(self)
        for qcnn in qcnns:
            new_qcnn = new_qcnn.merge(qcnn)
        return new_qcnn

    def update_Q(self, Q):
        """
        Update the number of available qubits for the qcnn and update the rest of the stack accordingly.

        Args:
            Q (list(int or string)): The list of available qubits.
        """
        motif = self.tail(Q)
        while motif.next is not None:
            motif = motif.next(motif.Q_avail)

    def __add__(self, other):
        """
        Add a motif, motifs or qcnn to the stack.

        Args:
            other (Qmotif, Qcnn, Sequence(Qmotif)): The motif, motifs or qcnn to add to the stack.

        Returns:
            Qcnn: A new Qcnn object with the motif(s) added to the stack.
        """
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
        """
        Repeat the qcnn a number of times. If a motif(s) is provided, it is added to the stack.

        Args:
            other (int, Qmotif, Sequence(Qmotif)): The number of times to repeat the qcnn or the motif(s) to add to the stack.

        Returns:
            Qcnn: A new Qcnn object with the motif(s) added to the stack.
        """
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
        """
        Generator to go from head to tail and only return operations (motifs that correspond to operations).
        """
        # Generator to go from head to tail and only return operations
        current = self.tail
        while current is not None:
            if current.is_operation:
                yield current
            current = current.next


class Qfree(Qmotif):
    """
    Qfree motif, represents a freeing up qubit for the QCNN, that is making qubits available for future operations. All Qcnn objects start with a Qfree motif.
    It is a special motif that has no edges and is not an operation.
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
        """
        Add a motif, motifs or qcnn to the stack with self.Qfree availabe qubits.
        """
        return Qcnn(self) + other

    def __call__(self, Q):
        """
        Calling Qfree doesn't do anything new, just returns the object.
        """
        self.set_Q(self.Q)
        self.set_Qavail(self.Q)
        return self
