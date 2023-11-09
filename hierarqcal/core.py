"""
This module contains the core classes for the hierarqcal package. :py:class:`Qmotif` is the base class for all primitives, it's a directed graph functioning as the building block for higher level motifs. :py:class:`Qhierarchy` is the full compute graph / architecture of the quantum circuit, it manages the interaction between motifs, their execution and symbol distribuition.

Create a hierarchical circuit as follows:

.. code-block:: python

    from hierarqcal import Qinit, Qcycle, Qmask
    hierq = Qinit(8) + (Qcycle(1) + Qmask("right")) * 3

The above creates a circuit that resembles a reverse binary tree architecture. There are 8 initial qubits and then a cycle-masking unit is repeated three times.
"""

from collections.abc import Sequence
from enum import Enum
import warnings
from copy import copy, deepcopy
from collections import deque, namedtuple
import numpy as np
import itertools as it
import sympy as sp

CircuitInstruction = namedtuple(
    "CircuitInstruction", ["gate_name", "symbol_info", "sub_bits"]
)


class Primitive_Types(Enum):
    """
    Enum for primitive types.
    """

    CYCLE = "cycle"
    MASK = "mask"
    SPLIT = "split"
    PERMUTE = "permute"
    INIT = "init"
    PIVOT = "pivot"
    BASE_MOTIF = "base_motif"


class Qunitary:
    """
    Base class for all unitary operations, the main purpose is to store the operation, its arity and parameters (symbols) for later use.
    # TODO add support for function as matrix
    # TODO add share_weights parameter
    """

    def __init__(self, function=None, n_symbols=0, arity=2, symbols=None, name=None):
        """
        Args:
            function (function, optional): Function to apply. If None, then the default from :py:class:`Default_Mappings` is used.
            n_symbols (int, optional): Number of symbols that function uses. Defaults to 0.
            arity (int, optional): Number of qubits that function acts upon. Two means 2-qubit unitaries, three means 3-qubits unitaries and so on. Defaults to 2.
            symbols (list, optional): List of symbol values (rotation angles). Elements can be either sympy symbols, complex numbers, or qiskit parameters. Defaults to None.
        """
        self.function = function
        if isinstance(self.function, str):
            (
                circuit_instructions,
                unique_bits,
                unique_params,
            ) = self.get_circ_info_from_string(function)
            self.circuit_instructions = circuit_instructions
            self.n_symbols = len(unique_params)
            self.arity = len(unique_bits)
        else:
            self.circuit_instructions = None
            self.n_symbols = len(symbols) if not (symbols is None) else n_symbols
            self.arity = arity
        self.symbols = symbols
        self.edge = None
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def get_symbols(self):
        """
        Get symbols for this unitary.

        Returns: List of symbols
        """
        return self.symbols

    def set_symbols(self, symbols=None):
        """
        Set symbols for this unitary.

        Args:
            symbols (list): List of symbols
        """

        if len(symbols) != self.n_symbols:
            raise ValueError(
                f"Number of symbols must be {self.n_symbols} for this function"
            )
        self.symbols = symbols

    def set_edge(self, edge):
        self.edge = edge

    def get_circ_info_from_string(self, input_str):
        """
        Takes a string that represents a circuit function, and
        breaks down the string into a set of circuit instructions.

        Args:
            `input_str` (str)
        Returns:
            `substr_list` (list): a list of circuit instructions, where each entry
            represents a distinct gate operation.
            Each entry is a list of three components: [gate_name,symbol_info, sub_bits]
                1. `gate_name` (str) is the name of the Qiskit gate being implemented.
                2. `symbol_info` (list) keeps track of whether the gate is parametrized, and
                        if so, whether it is the same parameter as another gate.
                3. `sub_bits` (list of ints) keeps track of the bits the gates are applied on.
            `unique_bits` (list of ints): set of qubits
            `unique_params` (list of strs): set of gate parameters

        Workflow:
            Step 1: partition the string into lists of individual gate instructions
                    in the form `{gate_string}(parameters)^{bits}`
            Step 2: split each substring into the gate string, the relevant
                    parameters, and the bits it acts on
            Step 3: convert the bits, the gate string, and the relevant parameters
                    into integers/functions
        """

        # Step 1 #

        # Split the input string based on ';' into a list where each entry is a gate instruction
        substrings = input_str.split(";")
        # Remove any leading or trailing whitespaces from each substring
        substrings = [substring.strip() for substring in substrings]

        # Steps 2,3 #

        circuit_instructions = []
        unique_bits = []
        unique_params = []
        for substring in substrings:
            instruction = []

            # Separating the parameters, gates, and bits in the substring
            start_index = substring.find("{")
            end_index = substring.find("}")

            param_start_index = substring.find("(")
            param_end_index = substring.find(")")

            bits_start_index = substring.find("^")
            bits_end_index = len(substring)

            # getting gate string
            if start_index == -1:
                gate_string = substring[start_index + 1 : param_start_index]
            else:
                gate_string = substring[start_index + 1 : end_index]
            instruction.append(gate_string.lower())

            # getting param string and index
            params_string = substring[param_start_index + 1 : param_end_index]
            if params_string == "":
                instruction.append([0, 0, 0])
            else:
                param_list = params_string.split(",")
                p_list = []
                p_inds = []
                for param_entry in param_list:
                    # Remove any leading or trailing whitespaces from each substring
                    p_entry = param_entry.strip()
                    p_list.append(p_entry)
                    # Check if the param is a duplicate and if not, add to unique_params
                    if p_entry not in unique_params:
                        unique_params.append(p_entry)
                        isinlist = False
                    else:
                        isinlist = True
                    # Find location of param in unique_params
                    param_indx = np.where(np.array(unique_params) == p_entry)[0][0]
                    p_inds.append(param_indx)

                instruction.append([len(p_list), p_inds, isinlist])

            # getting list of bits
            bits_string = substring[bits_start_index + 1 : bits_end_index]
            bits = []
            for bit in bits_string:
                bit = int(bit)
                bits.append(bit)
                if bit not in unique_bits:
                    unique_bits.append(bit)
            instruction.append(bits)

            # add to list of circuit instructions
            circuit_instruction = CircuitInstruction(
                instruction[0], instruction[1], instruction[2]
            )
            circuit_instructions.append(circuit_instruction)

        return circuit_instructions, unique_bits, unique_params


class Default_Mappings(Enum):
    """
    Enum for default mappings
    """

    CYCLE = Qunitary(n_symbols=1, arity=2)
    PIVOT = Qunitary(n_symbols=1, arity=2)
    MASK = None
    SPLIT = None
    BASE_MOTIF = None
    PERMUTE = Qunitary(n_symbols=1, arity=2)


class Qmotif:
    """
    Hierarchical circuit architectures are created by stacking motifs, the lowest level motifs (primitives) are building blocks for higher level ones. Examples of primitives are cycles (:py:class:`Qcycle`), masks (:py:class:`Qmask_Base`), and permutations (:py:class:`Qpermute`). Each motif is a directed graph with nodes Q representing qubits and edges E unitary operations applied between them. The direction of an edge is the order of interaction for the unitary. Each instance has pointers to its predecessor and successor.

    Attributes:
        Q (list, optional): Qubit labels of the motif. Defaults to [].
        E (list, optional): Edges of the motif. Defaults to [].
        Q_avail (list, optional): Available qubits of the motif. This is calculated by the previous motif in the stack. Defaults to [].
        edge_order (list, optional): Order of unitaries applied. Defaults to [1], [-1] will reverse the order, [5,6] does the 5th and 6th edge first and then the rest in the normal order.
        next (:py:class:`Qmotif`, optional): Next motif in the stack. Defaults to None.
        prev (Qmotif, optional): Previous motif in the stack. Defaults to None.
        mapping (:py:class:`Qunitary` or :py:class:`Qhierarchy`, optional): Either a :py:class:`Qunitary` instance or a :py:class:`Qhierarchy` that will be converted to an :py:class:`Qunitary` object. Defaults to None.
        symbol_fn (lambda): TODO
        is_default_mapping (bool, optional): Flag to determine if default mapping is used. Defaults to True.
        is_operation (bool, optional): Flag to determine if the motif is an operation. Defaults to True.
        share_weights (bool, optional): Flag to determine if weights are shared within a motif. Defaults to True.
    """

    def __init__(
        self,
        Q=[],
        E=[],
        Q_avail=[],
        edge_order=[1],
        next=None,
        prev=None,
        mapping=None,
        symbol_fn=lambda x, ns, ne: x,
        is_default_mapping=True,
        is_operation=True,
        share_weights=True,
        type=Primitive_Types.BASE_MOTIF,
    ) -> None:
        # Meta information
        self.is_operation = is_operation
        # self.is_default_mapping = is_default_mapping
        self.is_default_mapping = True if mapping is None else False
        self.type = type
        # graph
        self.Q = Q
        self.Q_avail = Q_avail
        self.E = E
        self.edge_order = edge_order
        # higher level graph
        self.mapping = mapping
        self.symbol_fn = symbol_fn
        self.share_weights = share_weights
        self.edge_mapping = (
            []
        )  # TODO docstring, edgemapping only gets set when the motif gets edges generated
        self.n_symbols = 0
        # pointers
        self.prev = prev
        self.next = next
        # Handle mappings
        if self.is_default_mapping:
            self.arity = 2
        else:
            if isinstance(self.mapping, Qhierarchy):
                # If mapping was specified as sub-hierarchy, convert it to a qunitary
                if any(
                    [
                        isinstance(symbol, sp.Symbol)
                        for symbol in self.mapping.get_symbols()
                    ]
                ):
                    # if any symbols are symbolic, then we scrap them TODO ensure if we want to do this
                    new_symbols = None
                else:
                    # If they are numeric
                    new_symbols = list(self.mapping.get_symbols())
                new_mapping = Qunitary(
                    function=None,
                    n_symbols=self.mapping.n_symbols,
                    arity=len(self.mapping.tail.Q),
                    symbols=new_symbols,
                    name=self.mapping.tail.name, # Qinit contains name
                )
                new_mapping.function = self.mapping.get_unitary_function()
                self.mapping = new_mapping
            self.arity = self.mapping.arity

    def __add__(self, other):
        """
        Append an other motif to the current one: self + other = (self, other).

        Args:
            other (Qmotif): Motif to append to current motif (self).

        Returns:
            Qmotifs: A 2-tuple of motifs: (self, other) in the order they were added. The tuples contain copies of the original motifs.
        """
        return self.append(other)

    def __mul__(self, other):
        """
        Repeat motif "other" times: self * other = (self, self, ..., self). Other is an int.

        Args:
            other (int): Number of times to repeat motif.

        Returns:
            Qmotifs: A tuple of motifs: (self, self, ..., self), where each is a new object copied from the original.
        """
        return Qmotifs((deepcopy(self) for i in range(other)))

    def __call__(self, Q, E=None, remaining_q=None, is_operation=True, **kwargs):
        self.set_Q(Q)
        mapping = kwargs.get("mapping", None)
        if mapping:
            self.set_mapping(mapping)
        if E is None:
            # If no new edge were provided
            q_old = kwargs.get("q_old", None)
            if q_old is not None:
                # If no new edges were provided and old qubits are present meaning qubit names changed
                E = [tuple(Q[q_old.index(i)] for i in e) for e in self.E]
            else:
                # No new edges and qubits didn't changed, so the motif edges stays the same
                if any([q not in self.Q for e in self.E for q in e]):
                    raise ValueError(
                        "Edge contains values not in qubit labels, Qmotif requires qubit labels (value not order) to be specified\nedge: {}\nqubit labels: {}".format(
                            self.E, self.Q
                        )
                    )

                else:
                    E = self.E
        self.set_E(E)
        if remaining_q:
            self.set_Qavail(remaining_q)
        else:
            self.set_Qavail(Q)
        self.set_is_operation(is_operation)
        start_idx = kwargs.get("start_idx", 0)
        self.set_symbols(start_idx=start_idx)
        return self

    def append(self, other):
        """
        Append an other motif to the current one: self + other = (self, other).

        Args:
            other (Qmotif): Motif to append to current motif (self).

        Returns:
            Qmotifs: A 2-tuple of motifs: (self, other) in the order they were added. The tuples contain copies of the original motifs.
        """
        if isinstance(other, Sequence):
            return Qmotifs((deepcopy(self),) + deepcopy(other))
        else:
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
            E (list(tuples)): List of edges, where each edge is a tuple of qubit labels (self.Q).
        """
        if 0 in self.edge_order:
            # Raise error if 0 is in edge order
            raise ValueError(
                "Edge order can't contain 0, as there is no 0th edge. Use 1 instead, edge order is based on ordering, that is [1] means first edge comes first [2,8] means second edge comes first, then 8th edge comes second. There is no 0th edge"
            )
        if -1 in self.edge_order:
            # Reverse edge order if -1 is in edge order
            Ep_l = [E[i] for i in range(len(E) - 1, -1, -1)]
        else:
            E_ordered = [E[i - 1] for i in self.edge_order if i - 1 < len(E)]
            E_rest = [edge for edge in E if edge not in E_ordered]
            Ep_l = E_ordered + E_rest
        self.E = Ep_l

    def set_arity(self, arity):
        """
        Set the arity of the motif (qubits per unitary/ nodes per edge).

        Args:
            arity (int): Number of qubits per unitary
        """
        self.arity = arity

    def set_edge_order(self, edge_order):
        """
        Set the edge order of the motif (order of unitaries applied). [1] means first edge comes first [2,8] means second edge comes first, then 8th edge comes second. For example, if self.E = [(1,2),(7,3),(5,2)] then self.edge_order = [1,2,3] means the unitaries are applied in the order (1,2),(7,3),(5,2). If self.edge_order = [2,1,3] then the unitaries are applied in the order (7,3),(1,2),(5,2). If self.edge_order = [3] then the unitaries are applied in the order (5,2),(1,2),(7,3),.

        Args:
            edge_order (list(int)): List of edge orders.
        """
        self.edge_order = edge_order

    def set_Qavail(self, Q_avail):
        """
        Set the number of available qubits Q_avail for the motif. This gets calculated by the previous motif in the stack, for example the :py:class:Qmask motif would mask qubits and use this function to update the available qubits after it's action. Example: if Q = [1,2,3,4] and Qmask("right") is applied then Q_avail = [1,2].

        Args:
            Q_avail (list): List of available qubits.
        """
        self.Q_avail = Q_avail

    def set_mapping(self, mapping):
        """
        Specify the unitary operations applied according to the type of motif.

        Args:
            mapping (Qhierarchy or Qunitary): Unitary operation applied to the motif.
        """
        self.mapping = mapping

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

    def set_share_weights(self, share_weights):
        """
        Set the share_weights flag.

        Args:
            share_weights (bool): Whether to share weights within a motif.
        """
        self.share_weights = share_weights

    def set_is_operation(self, is_operation):
        """
        Set the is_operation flag.

        Args:
            is_operation (bool): Whether the motif is an operation.
        """
        self.is_operation = is_operation

    def set_edge_mapping(self, unitary_function):
        """
        Maps each edge to a unitary function.

        Args:
            edge_mapping (function): function for each edge.
        """
        for unitary in self.edge_mapping:
            unitary.function = unitary_function

    def get_symbols(self):
        """
        Get the symbols of the motif. If share_weights is True, then symbols are obtained from the first edge_mapping. Otherwise, symbols are obtained from each edge_mapping.

        Yields:
            symbols (List) or None: List of symbols or None if no edge_mapping.
        """
        if len(self.edge_mapping) > 0:
            if self.share_weights:
                yield from self.edge_mapping[0].get_symbols()
            else:
                for edge_map in self.edge_mapping:
                    yield from edge_map.get_symbols()

    def set_symbols(self, symbols=None, start_idx=0):
        """
        Set the symbol's.

        Args:
            symbols (list): List of symbols to set.
            start_idx (int): Starting index of symbols, this is used when :py:class:Qhierarchy updates the stack, it loops through each motif counting symbols, at each motif it updates the symbols (inderectly calls this function) and send the current count as starting inde so that correct sympy symbol indices are used.
        """
        if not (self.mapping is None) and len(self.E) > 0:
            symbol_fn = self.symbol_fn
            if symbols is None:
                if (
                    isinstance(next(self.get_symbols(), False), sp.Symbol)
                    or len(list(self.get_symbols())) == 0
                ):
                    # If no new symbols are provided and current symbols are still symbolic or no symbols exist
                    # Generate symbols, this is used when Qhierarchy updates the stack.
                    if self.mapping.symbols is None:
                        # If no symbols exist in the mapping
                        symbols = sp.symbols(
                            f"x_{start_idx}:{start_idx + self.mapping.n_symbols*(len(self.E) if not(self.share_weights) else 1)}"
                        )
                    else:
                        symbols = self.mapping.symbols
                        if len(symbols) != self.mapping.n_symbols * (
                            len(self.E) if not (self.share_weights) else 1
                        ):
                            # If old symbols don't match current setup
                            symbols = [
                                symbol
                                for _ in range(
                                    (len(self.E) if not (self.share_weights) else 1)
                                )
                                for symbol in symbols
                            ]
                else:
                    # If no new symbols are provided but old symbols exist
                    symbols = list(self.get_symbols())
                    if len(symbols) != self.mapping.n_symbols * (
                        len(self.E) if not (self.share_weights) else 1
                    ):
                        # If old symbols don't match current setup
                        symbols = sp.symbols(
                            f"x_{start_idx}:{start_idx + self.mapping.n_symbols*(len(self.E) if not(self.share_weights) else 1)}"
                        )
                        raise Warning(
                            f"Number of symbols {len(symbols)} does not match number of symbols in motif {self.mapping.n_symbols*(len(self.E) if not(self.share_weights) else 1)}, symbolic ones will be generated"
                        )
            else:
                # Don't apply symbol fn (just identity) if symbols are set explicitly
                symbol_fn = lambda x, ns, ne: x
                if len(symbols) != self.mapping.n_symbols * (
                    len(self.E) if not (self.share_weights) else 1
                ):
                    raise ValueError(
                        f"Number of symbols {len(symbols)} does not match number of symbols in motif {self.mapping.n_symbols*(len(self.E) if not(self.share_weights) else 1)}"
                    )
            self.edge_mapping = []
            idx = 0
            n_edge = 1
            for edge in self.E:
                tmp_mapping = deepcopy(self.mapping)
                tmp_mapping.set_edge(edge)
                tmp_mapping.set_symbols(
                    [
                        symbol_fn(symbols[idx], idx + 1, n_edge)
                        for idx in range(idx, idx + self.mapping.n_symbols, 1)
                    ]
                )
                if not (self.share_weights):
                    idx += self.mapping.n_symbols
                self.edge_mapping.append(tmp_mapping)
                n_edge += 1
            self.n_symbols = len(symbols)

    def cycle(self, Q, stride=1, step=1, offset=0, boundary="periodic", arity=2):
        """
        The cycle pattern
        """
        nq_available = len(Q)
        if boundary == "open":
            mod_nq = lambda x: x % nq_available
            E = [
                tuple(
                    (
                        Q[i + j * stride]
                        for j in range(arity)
                        if i + j * stride < nq_available
                    )
                )
                for i in range(offset, nq_available, step)
            ]
            # Remove all that is not "complete"
            E = [edge for edge in E if len(edge) == arity]

        else:
            mod_nq = lambda x: x % nq_available
            E = [
                tuple((Q[mod_nq(i + j * stride)] for j in range(arity)))
                for i in range(offset, nq_available, step)
            ]
            # Remove all that is not "complete", i.e. contain duplicates
            E = [edge for edge in E if len(set(edge)) == arity]
        if (
            len(E) == arity
            and sum([len(set(E[0]) - set(E[k])) == 0 for k in range(arity)]) == arity
        ):
            # If there are only as many edges as qubits, and they are the same, then we can keep only one of them
            if arity > 0:
                E = [E[0]]
        return E


class Qmotifs(tuple):
    """
    A tuple of motifs, this is the data structure for storing sequences motifs. It subclasses tuple, so all tuple methods are available.
    """

    # TODO mention assumption that only operators should be used i.e. +, *
    # TODO add test to ensure the case (a,b)+b -> (a,b,c) no matter type of b
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


class Qcycle(Qmotif):
    """
    A cycle motif, spreads unitaries in a ladder structure across the circuit
    TODO implement open boundary for dense and pooling also + tests
    """

    def __init__(
        self,
        stride=1,
        step=1,
        offset=0,
        boundary="periodic",
        **kwargs,
    ):
        """
        TODO determine if boundary is the best name for this, or if it should be called something else.
        TODO docstring for edge order, importantly it's based on ordering, that is [1] means first edge comes first [2,8] means second edge comes first, then 8th edge comes second. There is no 0th edge
        Initialize a cycle motif.

        Args:
            stride (int, optional): Stride of the cycle. Defaults to 1.
            step (int, optional): Step of the cycle. Defaults to 1.
            offset (int, optional): Offset of the cycle. Defaults to 0.

        """
        self.stride = stride
        self.step = step
        self.offset = offset
        self.boundary = boundary
        # Specify sequence of gates:
        mapping = kwargs.get("mapping", None)
        is_default_mapping = True if mapping is None else False
        # motif_symbols = None # TODO maybe allow symbols to be intialised
        # Initialize graph
        super().__init__(
            is_default_mapping=is_default_mapping, type=Primitive_Types.CYCLE, **kwargs
        )

    def __call__(self, Qc_l, *args, **kwargs):
        """
        Call the motif, this generates the edges and qubits of the motif (directed graph) based on it's available qubits.
        Each time a motif in the stack changes, a loop runs through the stack from the beginning and calls each motif to update the graph (the available qubits, the edges etc).

        Args:
            Qc_l (list): List of available qubits.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments, such as:

                * mapping (tuple(function, int)): TODO update this docstring, it's not a tuple anymore.
                    Function mapping is specified as a tuple, where the first argument is a function and the second is the number of symbols it uses. A symbol here refers to an variational paramater for a quantum circuit, i.e. crz(theta, q0, q1) <- theta is a symbol for the gate.

        Returns:
            Qconv: Returns the updated version of itself, with correct nodes and edges.
        """
        # Determine cycle operation
        if self.stride % len(Qc_l) == 0:
            # TODO make this clear in documentation
            # warnings.warn(
            #     f"Stride and number of available qubits can't be the same, received:\nstride: {self.stride}\n available qubits:{nq_available}. Defaulting to stride of 1"
            # )
            self.stride = 1
        E = self.cycle(
            Qc_l,
            stride=self.stride,
            step=self.step,
            offset=self.offset,
            boundary=self.boundary,
            arity=self.arity,
        )
        updated_self = super().__call__(Qc_l, E=E, **kwargs)
        return updated_self

    def __eq__(self, other):
        if isinstance(other, Qcycle):
            self_attrs = vars(self)
            other_attrs = vars(other)
            for attr, value in self_attrs.items():
                if attr not in other_attrs or other_attrs[attr] != value:
                    return False

            return True
        return False


class Qpermute(Qmotif):
    """
    A dense motif, it connects unitaries to all possible combinations of qubits (all possible edges given Q) in the quantum circuit.
    """

    def __init__(self, combinations=True, **kwargs):
        self.combinations = combinations
        # Specify sequence of gates:
        mapping = kwargs.get("mapping", None)
        is_default_mapping = True if mapping is None else False
        # Initialize graph
        super().__init__(
            is_default_mapping=is_default_mapping,
            type=Primitive_Types.PERMUTE,
            **kwargs,
        )

    def __call__(self, Qc_l, *args, **kwargs):
        """
        Call the motif, this is used to generate the edges and qubits of the motif (directed graph) based on it's available qubits.
        Each time a motif in the stack changes, a loop runs through the stack from the beginning and calls each motif to update the graph (the available qubits, the edges etc).

        Args:
            Qc_l (list): List of available qubits.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments, such as:

                * mapping (tuple(function, int)):
                    Function mapping is specified as a tuple, where the first argument is a function and the second is the number of symbols it uses. A symbol here refers to an variational paramater for a quantum circuit, i.e. crz(theta, q0, q1) <- theta is a symbol for the gate.

        Returns:
            Qdense: Returns the updated version of itself, with correct nodes and edges.
        """
        # All possible wire combinations
        if self.combinations:
            Ec_l = list(it.combinations(Qc_l, r=self.arity))
        else:
            Ec_l = list(it.permutations(Qc_l, r=self.arity))
        if len(Ec_l) == 2 and Ec_l[0][0:] == Ec_l[1][1::-1]:
            Ec_l = [Ec_l[0]]
        updated_self = super().__call__(Qc_l, E=Ec_l, **kwargs)
        return updated_self

    def __eq__(self, other):
        if isinstance(other, Qpermute):
            self_attrs = vars(self)
            other_attrs = vars(other)
            for attr, value in self_attrs.items():
                if attr not in other_attrs or other_attrs[attr] != value:
                    return False
            return True
        return False


class Qsplit(Qmotif):
    def __init__(
        self,
        global_pattern="1*",
        merge_within="1*",
        merge_between=None,  # either this or third item in strides
        mask=False,
        strides=[1, 1, 0],
        steps=[1, 1, 1],
        offsets=[0, 0, 0],
        boundaries=["open", "open", "periodic"],
        type=Primitive_Types.SPLIT,
        **kwargs,
    ) -> None:
        # If strides, steps or offsets are provided as integers, convert to list that repeats that integer
        if isinstance(strides, int):
            strides = [strides] * 3
        if isinstance(strides, int):
            steps = [steps] * 3
        if isinstance(offsets, int):
            offsets = [offsets] * 3
        # Set attributes

        self.global_pattern = global_pattern
        self.merge_within = merge_within
        self.merge_between = merge_between
        self.mask = mask
        self.strides = strides
        self.steps = steps
        self.offsets = offsets
        self.boundaries = boundaries
        mapping = kwargs.get("mapping", None)
        is_default_mapping = True if mapping is None else False
        # Initialize Qmotif
        super().__init__(is_default_mapping=is_default_mapping, type=type, **kwargs)

    def __call__(self, Q, E=[], remaining_q=None, is_operation=True, **kwargs):
        updated_self = super().__call__(
            Q, E=E, remaining_q=remaining_q, is_operation=is_operation, **kwargs
        )
        return updated_self

    def wildcard_populate(self, pattern, length):
        # Wildcard pattern
        n_stars = pattern.count("*")
        n_excl = pattern.count("!")
        # base = pattern.replace("*", "0" * zero_per_star)
        # base = base.replace("!", "1" * zero_per_star)
        base = pattern
        max_it = length
        do_star = True if n_stars > 0 else False
        just_changed = False
        stars_found = 0
        excls_found = 0
        while len(base) < (length + n_stars + n_excl) and max_it > 0:
            """
            TODO refactor this code so that it is more readable. The idea is this:
            There are two possible wild cards, excl: ! and star: *. ! fills with 1's and * fills with 0's. We want to distribute 0's and 1's as evenly as we can based on the provided pattern. Some examples, if we have 8 qubits:
            1*1 -> 10000001
            0!0 -> 01111110
            *! -> 00001111
            *1* -> 00001000
            The algorithm alternates between finding a star and an excl and inserts a 0 or 1 next to it. The first iteration checks for a star (since we need to pick one), therefore it has a kind of precedence, but only effects the first insertion. From there it alternates between star and excl.

            We have another alternation which is between using find and rfind, this is to handle the case when there's 2 stars or 2 exclamations. I don't think 3 wild cards make sense TODO think about this.
            """
            if do_star:
                if stars_found % 2 == 0:
                    idx = base.find("*")
                else:
                    idx = base.rfind("*")
                stars_found += 1
                # Insert 0 next to it
                base = base[:idx] + "0" + base[idx:]
                do_star = False if n_excl > 0 else True
                just_changed = True
            if (not do_star) and (not just_changed):
                if excls_found % 2 == 0:
                    idx = base.find("!")
                else:
                    idx = base.rfind("!")
                excls_found += 1
                # Insert 1 next to it
                base = base[:idx] + "1" + base[idx:]
                do_star = True if n_stars > 0 else False
            just_changed = False
            max_it -= 1
        base = base.replace("*", "")
        base = base.replace("!", "")
        return base

    def get_pattern_fn(self, pattern, length):
        # If pattern is a string then convert it to a lambda function
        if isinstance(pattern, str):
            # If pattern contains wild cards, then we need to populate it
            if any(("*" == c) or ("!" == c) for c in pattern):
                pattern = self.wildcard_populate(pattern, length)
            if len(pattern) < length:
                # If there are no wildcard characters, then we assume that the pattern is a base pattern and we will repeat it until it is the same length as the current number of qubits
                base = pattern * (length // len(pattern))
                pattern = base[:length]
            # Pattern is now a string of 1's and 0's and have length >= lengthor some predefined string
            if all(c in ["0", "1"] for c in pattern):
                pattern_fn = lambda arr: [
                    item for item, indicator in zip(arr, pattern) if indicator == "1"
                ]
            else:
                # Pattern must be predefined string
                pattern_fn = self.get_predefined_pattern_fn(pattern)
        # If pattern is already in lambda function form, then just use it
        else:
            # Check if pattern is function
            if callable(pattern):
                pattern_fn = pattern
            else:
                raise Exception("Pattern must be a string or a lambda function")
        return pattern_fn

    def cycle_between_splits(
        self, E_a, E_b, stride=0, step=1, offset=0, boundary="open"
    ):
        if boundary == "open":
            E = [
                (
                    E_a[i],
                    E_b[(i + stride)],
                )
                for i in range(offset, len(E_a), step)
                if (i + stride) < len(E_b)
            ]
        elif boundary == "periodic":
            E = [
                (
                    E_a[i],
                    E_b[(i + stride) % len(E_b)],
                )
                for i in range(offset, len(E_a), step)
            ]
        else:
            raise Exception("Boundary must be either open or periodic")
        return E

    def merge_within_splits(self, E, merge_pattern):
        E_out = []
        for e in E:
            i_0 = 0
            i_1 = 0
            dummy = tuple()
            for char in merge_pattern:
                if char == "1":
                    dummy += (e[0][i_0],)
                    i_0 += 1
                elif char == "0":
                    dummy += (e[1][i_1],)
                    i_1 += 1
                else:
                    raise Exception("Merge pattern must be a string of 0's and 1's")
            E_out.append(dummy)
        return E_out

    def get_predefined_pattern_fn(self, pattern):
        # Mapping words to the pattern type
        if pattern == "left":
            # 0 1 2 3 4 5 6 7
            # x x x x
            pattern_fn = lambda arr: arr[0 : len(arr) // 2 : 1]
        elif pattern == "right":
            # 0 1 2 3 4 5 6 7
            #         x x x x
            pattern_fn = lambda arr: arr[len(arr) // 2 : len(arr) : 1]
        elif pattern == "even":
            # 0 1 2 3 4 5 6 7
            # x   x   x   x
            pattern_fn = lambda arr: arr[0::2]
        elif pattern == "odd":
            # 0 1 2 3 4 5 6 7
            #   x   x   x   x
            pattern_fn = lambda arr: arr[1::2]
        elif pattern == "inside":
            # 0 1 2 3 4 5 6 7
            #     x x x x
            pattern_fn = (
                lambda arr: arr[
                    len(arr) // 2 - len(arr) // 4 : len(arr) // 2 + len(arr) // 4 : 1
                ]
                if len(arr) > 2
                else [arr[1]]
            )  # inside
        elif pattern == "outside":
            # 0 1 2 3 4 5 6 7
            # x x         x x
            pattern_fn = (
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
            raise ValueError(f"{pattern} - Pattern not recognized")
        return pattern_fn


class Qmask(Qsplit):
    """
    A masking motif, it masks qubits based on some pattern TODO some controlled operation where the control is not used for the rest of the circuit).
    This motif changes the available qubits for the next motif in the stack.
    """

    def __init__(
        self,
        global_pattern="1*",
        merge_within="1*",
        merge_between=None,
        strides=[1, 1, 0],
        steps=[1, 1, 1],
        offsets=[0, 0, 0],
        boundaries=["open", "open", "periodic"],
        **kwargs,
    ):
        """
        TODO allow strides,steps,offsets to be one value and repeat what was given
        """
        if isinstance(strides, int):
            strides = [strides] * 3
        if isinstance(strides, int):
            steps = [steps] * 3
        if isinstance(offsets, int):
            offsets = [offsets] * 3
        super().__init__(
            global_pattern=global_pattern,
            merge_within=merge_within,
            merge_between=merge_between,
            strides=strides,
            steps=steps,
            offsets=offsets,
            boundaries=boundaries,
            mask=True,
            type=Primitive_Types.MASK,
            **kwargs,
        )

    def __call__(self, Qp_l, *args, **kwargs):
        """
        Call the motif, this is used to generate the edges and qubits of the motif (directed graph) based on it's available qubits.
        Each time a motif in the stack changes, a loop runs through the stack from the beginning and calls each motif to update the graph (the available qubits, the edges etc).

        Args:
            Qp_l (list): List of available qubits.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments, such as:

                * mapping (tuple(function, int)):
                    Function mapping is specified as a tuple, where the first argument is a function and the second is the number of symbols it uses. A symbol here refers to an variational paramater for a quantum circuit, i.e. crz(theta, q0, q1) <- theta is a symbol for the gate.

        Returns:
            Qpool: Returns the updated version of itself, with correct nodes and edges.
        """
        # The idea is to mask qubits based on some pattern
        # This can be done with or without applying a unitary. Applying a unitary "preserves" their information (usually through some controlled unitary)
        # This enables patterns of: pooling in quantum neural networks, coarse graining or entanglers or just plain masking
        # Default is mask without a mapping, making it non operational
        is_operation = False
        # Defaults for when nothing happens (this gets changed if conditions are met, i.e. there are qubits to mask etc)
        Ep_l = []
        remaining_q = Qp_l
        # If there are qubits to mask
        if len(Qp_l) > 1:
            # Get global pattern function based on the pattern attribute
            self.mask_pattern_fn = self.get_pattern_fn(self.global_pattern, len(Qp_l))
            # Apply pattern function on all available qubits
            measured_q = self.mask_pattern_fn(Qp_l)
            remaining_q = [q for q in Qp_l if not (q in measured_q)]
            if len(remaining_q) == 0:
                # Don't do anything if all qubits were removed
                remaining_q = Qp_l
            elif not (self.mapping is None):
                # there is a operation associated with the motif
                is_operation = True
                # Populate merge pattern
                merge_within_pop = self.wildcard_populate(self.merge_within, self.arity)
                # Count the number of 1s in the merge pattern
                arity_m = merge_within_pop.count("1")
                arity_r = self.arity - arity_m
                # Generate edges for measured split
                E_m = self.cycle(
                    measured_q,
                    stride=self.strides[0],
                    step=self.steps[0],
                    offset=self.offsets[0],
                    boundary=self.boundaries[0],
                    arity=arity_m,
                )
                # Generate edges for remaining split
                E_r = self.cycle(
                    remaining_q,
                    stride=self.strides[1],
                    step=self.steps[1],
                    offset=self.offsets[1],
                    boundary=self.boundaries[1],
                    arity=arity_r,
                )
                # Generate edges for measured to remaining
                if not (self.merge_between == None):
                    # If there is a merge_between pattern
                    pattern_fn = self.get_pattern_fn(self.merge_between, len(E_r))
                    E_r = pattern_fn(E_r)
                if len(E_m) > 0 and len(E_r) > 0:
                    E_b = self.cycle_between_splits(
                        E_a=E_m,
                        E_b=E_r,
                        stride=self.strides[2],
                        step=self.steps[2],
                        offset=self.offsets[2],
                        boundary=self.boundaries[2],
                    )
                    # Merge the two splits based on merge pattern
                    Ep_l = self.merge_within_splits(E_b, merge_within_pop)
                else:
                    # Do nothing if Em or Er was empty
                    remaining_q = Qp_l
                    Ep_l = []

        updated_self = super().__call__(
            Qp_l, E=Ep_l, remaining_q=remaining_q, is_operation=is_operation, **kwargs
        )
        return updated_self

    def __eq__(self, other):
        if isinstance(other, Qmask):
            self_attrs = vars(self)
            other_attrs = vars(other)

            for attr, value in self_attrs.items():
                if not (attr == "mask_pattern_fn"):
                    if attr not in other_attrs or other_attrs[attr] != value:
                        return False

            return True
        return False


class Qunmask(Qsplit):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        TODO possibility to give masking motif to undo
        """
        super().__init__(*args, type=Primitive_Types.MASK, **kwargs)

    def __call__(self, Qp_l, *args, **kwargs):
        """
        TODO
        """
        if self.global_pattern == "previous":
            current = self
            unmasked_q = []
            unmask_counts = 0
            q_old = kwargs.get("q_initial", [])
            while current is not None:
                current = current.prev
                if isinstance(current, Qmask) and unmask_counts <= 0:
                    unmasked_q = current.Q
                    q_old = current.Q
                    current = None

                if isinstance(current, Qunmask):
                    unmask_counts += 1
                if isinstance(current, Qmask):
                    unmask_counts -= 1
        else:
            q_old = kwargs.get("q_initial", [])
            self.mask_pattern_fn = self.get_pattern_fn(self.global_pattern, len(q_old))
            unmasked_q = self.mask_pattern_fn(q_old)
        is_operation = False
        Ep_l = []
        unique_unmasked = [q for q in unmasked_q if q not in Qp_l]
        new_avail_q = [q for q in q_old if q in Qp_l + unique_unmasked]
        updated_self = super().__call__(
            Qp_l, E=Ep_l, remaining_q=new_avail_q, is_operation=is_operation, **kwargs
        )
        return updated_self


class Qpivot(Qsplit):
    """
    The pivot connects the set of available qubits sequentially to a fixed set of qubits. The global pattern determine the pivot points while the merge pattern determines how the qubits are passed to the mapping.
    """

    def __init__(
        self,
        global_pattern="1*",
        merge_within="1*",
        merge_between=None,
        strides=[1, 1, 0],
        steps=[1, 1, 1],
        offsets=[0, 0, 0],
        boundaries=["open", "open", "periodic"],
        **kwargs,
    ):
        """
        Allow strides,steps,offsets and boundaries to be one value and repeat what was given
        """
        if isinstance(strides, int):
            strides = [strides] * 3
        if isinstance(strides, int):
            steps = [steps] * 3
        if isinstance(offsets, int):
            offsets = [offsets] * 3
        if isinstance(boundaries, str):
            offsets = [boundaries] * 3

        super().__init__(
            global_pattern=global_pattern,
            merge_within=merge_within,
            merge_between=merge_between,
            strides=strides,
            steps=steps,
            offsets=offsets,
            boundaries=boundaries,
            mask=False,
            type=Primitive_Types.PIVOT,
            **kwargs,
        )

    def __call__(self, Qp_l, *args, **kwargs):
        """

        Args:
            Qp_l (list): List of available qubits.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments, such as:
                * mapping (tuple(function, int)):
                    Function mapping is specified as a tuple, where the first argument is a function and the second is the number of symbols it uses. A symbol here refers to an variational paramater for a quantum circuit, i.e. crz(theta, q0, q1) <- theta is a symbol for the gate.

        Returns:
        """

        # Pivot must have a mapping
        # TODO add a default mapping?
        if self.mapping is None:
            raise Exception("Pivot must have a mapping")

        # Count the number of 1s in the merge pattern
        arity_p = self.merge_within.count("1")
        arity_r = self.arity - arity_p

        # Get global pattern function based on the pattern attribute
        self.pivot_pattern_fn = self.get_pattern_fn(
            self.global_pattern.replace("1", "1" * arity_p), len(Qp_l)
        )
        pivot_q = [
            p
            for i in range((len(self.pivot_pattern_fn(Qp_l)) + arity_p - 1) // arity_p)
            for p in self.pivot_pattern_fn(Qp_l)[i * arity_p : (i + 1) * arity_p]
        ]

        remaining_q = [q for q in Qp_l if not (q in pivot_q)]

        E_p = self.cycle(
            pivot_q,
            stride=self.strides[0],
            step=self.steps[0],
            offset=self.offsets[0],
            boundary=self.boundaries[0],
            arity=arity_p,
        )

        if arity_r > 0:
            # Generate edges for remaining split
            E_r = self.cycle(
                remaining_q,
                stride=self.strides[1],
                step=self.steps[1],
                offset=self.offsets[1],
                boundary=self.boundaries[1],
                arity=arity_r,
            )

            # If E_r empty then there were not enough qubits to satisfy the arity
            if len(E_r) > 0 and len(E_p) > 0:
                # Duplicate items in E_p to match length of E_r such that each unique item in E_p can be matched to an equal number of items in E_r
                max_it = 0  # prevent infinite loop
                E_tmp = E_p.copy()
                N = len(E_r)
                while len(E_tmp + E_p) <= N and max_it < N:
                    E_tmp += E_p
                    max_it += 1
                E_p = E_tmp.copy()

                # Reorder E_p so that like pivots are grouped together, i.e. remaining available qubits are assigned first to the first pivot point, then the second and so on.
                E_tmp = []
                for i in range(N):
                    E_tmp += E_p[
                        i :: N + 1
                    ]  # TODO check if this change works as intended it used to be just N
                E_p = E_tmp.copy()

                # TODO what could merge_between be used for?
                if not (self.merge_between == None):
                    pass

                E_b = self.cycle_between_splits(
                    E_a=E_r,
                    E_b=E_p,
                    stride=self.strides[2],
                    step=self.steps[2],
                    offset=self.offsets[2],
                    boundary=self.boundaries[2],
                )

                E_b = [(e[1], e[0]) for e in E_b]

                # Merge the two splits based on merge pattern
                merge_within_pop = self.wildcard_populate(self.merge_within, self.arity)
                Ep_l = self.merge_within_splits(E_b, merge_within_pop)
            else:
                Ep_l = []
        else:
            Ep_l = E_p

        updated_self = super().__call__(
            Qp_l, E=Ep_l, remaining_q=Qp_l, is_operation=True, **kwargs
        )
        return updated_self

    def cycle_between_splits(
        self, E_a, E_b, stride=0, step=1, offset=0, boundary="open"
    ):
        if boundary == "open":
            E = [
                (
                    E_a[i],
                    E_b[(i + stride)],
                )
                for i in range(offset, len(E_a), step)
                if (i + stride) < len(E_b)
            ]
        elif boundary == "periodic":
            E = [
                (
                    E_a[i],
                    E_b[(i + stride) % len(E_b)],
                )
                for i in range(offset, len(E_a), step)
            ]
        else:
            raise Exception("Boundary must be either open or periodic")

        return E

    def __eq__(self, other):
        if isinstance(other, Qmask):
            self_attrs = vars(self)
            other_attrs = vars(other)

            for attr, value in self_attrs.items():
                if not (attr == "mask_pattern_fn"):
                    if attr not in other_attrs or other_attrs[attr] != value:
                        return False

            return True
        return False


class Qhierarchy:
    """
    The main class that manages the "stack" of motifs, it handles the interaction between successive motifs and when a motifs are added to the stack,
    it updates all the others accordingly. An object of this class fully captures the architectural information of a hierarchical quantum circuit.
    It also handles function (unitary operation) mappings.
    """

    def __init__(self, qubits, function_mappings={}) -> None:
        # Set available qubit
        if isinstance(qubits, Qmotif):
            self.tail = qubits
            self.head = self.tail
        else:
            self.tail = Qinit(qubits)
            self.head = self.tail
        self.function_mappings = function_mappings
        self.mapping_counter = {
            primitive_type.value: 1 for primitive_type in Primitive_Types
        }
        self.n_symbols = 0

    def __add__(self, other):
        """
        Add a motif, motifs or hierarchy to the stack.

        Args:
            other (Qmotif, Qhierarchy, Sequence(Qmotif)): The motif, motifs or hierarchy to add to the stack.

        Returns:
            Qhierarchy: A new Qhierarchy object with the motif(s) added to the stack.
        """
        if isinstance(other, Qhierarchy):
            new_qcnn = self.merge(other)
        elif isinstance(other, Sequence):
            if isinstance(other[-1], Qmotif):
                new_qcnn = self.extend(other)
            elif isinstance(other[-1], Qhierarchy):
                new_qcnn = self.extmerge(other)
        elif isinstance(other, Qmotif):
            new_qcnn = self.append(other)
        return new_qcnn

    def __mul__(self, other):
        """
        Repeat the hierarchy a number of times. If a motif(s) is provided, it is added to the stack.

        Args:
            other (int, Qmotif, Sequence(Qmotif)): The number of times to repeat the hierarchy or the motif(s) to add to the stack.

        Returns:
            Qhierarchy: A new Qhierarchy object with the motif(s) added to the stack.
        """
        # TODO
        if isinstance(other, Qhierarchy):
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

    def __iter__(
        self,
    ):
        """
        Generator to go from head to tail and only return operations (motifs that correspond to operations).
        """
        # Generator to go from head to tail and only return operations
        current = self.tail
        while current is not None:
            if current.is_operation:
                yield current
            current = current.next

    def __len__(self):
        """
        Returns the number of motifs in the hierarchy.
        """
        k = 0
        current = self.tail
        while current is not None:
            k += 1
            current = current.next
        return k

    def __repr__(self) -> str:
        description = ""
        current = self.tail
        while current is not None:
            if current.is_operation:
                description += f"{repr(vars(current))}\n"
            current = current.next
        return description

    def __eq__(self, other):
        if isinstance(other, Qhierarchy):
            for layer_self, layer_other in zip(self, other):
                if not (layer_self == layer_other):
                    return False
            return True
        return False

    def __getitem__(self, index):
        current = self.tail
        for i in range(index + 1):
            if i == index:
                return current
            current = current.next
        return None

    def __call__(self, symbols=None, backend=None, **kwargs):
        if backend == "pennylane":
            from hierarqcal.pennylane import execute_circuit_pennylane

            return execute_circuit_pennylane(self, symbols, **kwargs)
        elif backend == "qiskit":
            from hierarqcal.qiskit import get_circuit_qiskit

            return get_circuit_qiskit(self, symbols, **kwargs)
        elif backend == "cirq":
            from hierarqcal.cirq import get_circuit_cirq

            return get_circuit_cirq(self, symbols, **kwargs)
        else:
            if not (symbols is None):
                self.set_symbols(symbols)
            # Default backend
            # TODO set default mapping
            state = self.tail(self.tail.Q).state
            for layer in self:
                for unitary in layer.edge_mapping:
                    state = unitary.function(
                        bits=unitary.edge,
                        symbols=unitary.symbols,
                        state=state,
                    )
            return state

    def get_symbols(self):
        return (symbol for layer in self for symbol in layer.get_symbols())

    def set_symbols(self, symbols):
        idx = 0
        for layer in self:
            n_symbols = len([_ for _ in layer.get_symbols()])
            layer.set_symbols(symbols[idx : idx + n_symbols])
            idx += n_symbols

    def get_unitary_function(self, **kwargs):
        """
        Convert the Qhierarchy into a function that can be called.
        """

        def unitary_function(bits, symbols=None, **kwargs):
            self.update_Q(bits)
            if not (symbols is None):
                self.set_symbols(symbols)
            state = None
            for layer in self:
                for unitary in layer.edge_mapping:
                    if isinstance(unitary.function, str):
                        get_circuit_from_string = kwargs.get(
                            "get_circuit_from_string", None
                        )
                        unitary = get_circuit_from_string(unitary)
                    state = unitary.function(
                        unitary.edge, unitary.symbols, **kwargs
                    )
                    if kwargs.get("state", None) is not None:
                        kwargs["state"] = state
            return state

        return unitary_function

    def append(self, motif):
        """
        Add a motif to the stack of motifs and update it (call to generate nodes and edges) according to the action of the previous motifs in the stack.

        Args:
            motif (Qmotif): The motif to add to the stack.

        Returns:
            Qhierarchy: A new Qhierarchy object with the new motif added to the stack.
        """
        motif = deepcopy(motif)
        # old_head = deepcopy(self.head) TODO test
        self.head.set_next(motif)
        self.head.next.set_prev(self.head)
        self.head = self.head.next

        if motif.is_operation & motif.is_default_mapping:
            mapping = None
            mappings = self.function_mappings.get(motif.type, None)
            if mappings:
                # If function mapping was provided
                mapping = mappings[
                    (self.mapping_counter.get(motif.type) - 1) % len(mappings)
                ]
                self.mapping_counter.update(
                    {motif.type: self.mapping_counter.get(motif.type) + 1}
                )
                motif.is_default_mapping = False
            else:
                # If no function mapping was provided
                mapping = getattr(Default_Mappings, motif.type.name).value
            self.head(self.head.prev.Q_avail, mapping=mapping, q_initial=self.tail.Q)
            # self.update_symbols(motif) TODO

        else:
            n_symbols = len([_ for _ in self.get_symbols()])
            self.head(
                self.head.prev.Q_avail, start_idx=n_symbols, q_initial=self.tail.Q
            )
            # self.update_symbols(motif) TODO

        self.n_symbols = len([_ for _ in self.get_symbols()])
        return self

    def extend(self, motifs):
        """
        Add and update a list of motifs to the current stack of motifs call each to generate their nodes and edges according to the action of the previous motifs in the stack.

        Args:
            motifs (Qmotifs): A tuple of motifs to add to the stack.

        Returns:
            Qhierarchy: A new Qhierarchy object with the motifs added to the stack.
        """
        new_hierarchy = deepcopy(self)
        for motif in motifs:
            new_hierarchy = new_hierarchy.append(motif)
        return new_hierarchy

    def merge(self, hierarchy):
        """
        Merge two Qhierarchy objects by adding the head of the second one to the tail of the first one. Both are copied to ensure immutability.

        Args:
            hierarchy (Qhierarchy): The Qhierarchy object to merge with the current one.

        Returns:
            Qhierarchy: A new Qhierarchy object with the two merged.
        """
        other_hierarchy = deepcopy(hierarchy)
        new_hierarchy = deepcopy(self)
        other_hierarchy.update_Q(
            new_hierarchy.head.Q_avail, start_idx=new_hierarchy.n_symbols
        )
        new_hierarchy.head.set_next(other_hierarchy.tail)
        new_hierarchy.head = other_hierarchy.head
        return new_hierarchy

    def extmerge(self, hierarchies):
        """
        Merge a list of Qhierarchy objects by adding the head of each one to the tail of the previous one. All are copied to ensure immutability.

        Args:
            hierarchies (Qhierarchy): A list of Qhierarchy objects to merge with the current one.

        Returns:
            Qhierarchy: A new Qhierarchy object with the list merged.
        """
        new_qcnn = deepcopy(self)
        for hierarchy in hierarchies:
            new_qcnn = new_qcnn.merge(hierarchy)
        return new_qcnn

    def reverse(self):
        """
        Reverse the stack of motifs on the highest level, i.e. only the direct children nothing deeper.
        """
        old = self.head
        current = deepcopy(self.tail)
        while not (isinstance(old, Qinit)):
            current = current + old
            old = old.prev
        current.head.set_next(None)

        return current

    def replace(self, motif, selected_index):
        top_level_indices = range(1, len(self), 1)
        current = deepcopy(self[0])
        for index in top_level_indices:
            if index == selected_index:
                current = current + motif
            else:
                current = current + self[index]
        return current

    def update_Q(self, Q, start_idx=0):
        """
        Update the number of available qubits for the hierarchy and update the rest of the stack accordingly.

        Args:
            Q (list(int or string)): The list of available qubits.
        """
        q_old = self.tail.Q
        motif = self.tail(Q)
        while motif.next is not None:
            motif = motif.next(
                motif.Q_avail, start_idx=start_idx, q_initial=self.tail.Q, q_old=q_old
            )
            start_idx += motif.n_symbols

    def copy(self):
        """
        Returns:
            Qhierarchy: A copy of the current Qhierarchy object.
        """
        return deepcopy(self)


class Qinit(Qmotif):
    """
    Qinit motif, represents a freeing up qubit for the QCNN, that is making qubits available for future operations. All Qhierarchy objects start with a Qinit motif.
    It is a special motif that has no edges and is not an operation.
    """

    def __init__(
        self, Q, state=None, tensors=None, name=None, **kwargs
    ) -> None:
        if isinstance(Q, Sequence):
            Qinit = Q
        elif type(Q) == int:
            Qinit = [i for i in range(Q)]
        self.state = state
        self.tensors = tensors
        self.name = name
        # Initialize graph
        super().__init__(
            Q=Qinit,
            Q_avail=Qinit,
            is_operation=False,
            type=Primitive_Types.INIT,
            **kwargs,
        )

    def __add__(self, other):
        """
        Add a motif, motifs or hierarchy to the stack with self.Qinit available qubits.
        """
        return Qhierarchy(self) + other

    def __call__(self, Q, *args, **kwargs):
        """
        Calling TODO add explanation just returns the object. Kwargs and Args are ignored, it just ensures that Qinit can be called the same way operational motifs can.
        """
        if self.tensors is not None:
            """
            If tensors are provided, the state is initialized as the tensor product of all the tensors.
            """
            dimensions = [len(self.tensors[0])]
            state = self.tensors[0]
            for tensor in self.tensors[1:]:
                state = np.array(np.kron(state, tensor))
                dimensions += [len(tensor)]
            self.dimensions = dimensions
            self.state = state.reshape(dimensions)
        self.set_Q(Q)
        self.set_Qavail(Q)
        return self
