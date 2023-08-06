"""
Helper functions for qiskit
"""
import numpy as np
import sympy as sp
from hierarqcal.core import Primitive_Types
from .qiskit_circuits import U2, V2
import warnings
from qiskit.circuit import Parameter, QuantumCircuit, QuantumRegister


def get_qiskit_default_unitary(layer):
    if layer.type in [
        Primitive_Types.CYCLE,
        Primitive_Types.PERMUTE,
        Primitive_Types.PIVOT,
    ]:
        unitary_function = U2
    elif layer.type in [Primitive_Types.MASK]:
        unitary_function = V2
    else:
        warnings.warn(
            f"No default function mapping for primitive type: {layer.type}, please provide a mapping manually"
        )
    # Give all edge mappings correct default unitary
    return unitary_function


def get_circuit_qiskit(hierq, symbols=None, barriers=True):
    """
    The main helper function for qiskit, it takes a qcnn(:py:class:`hierarqcal.core.Qcnn`) object that describes the cicruit architecture
    and builds a qiskit.QuantumCircuit object with the correct function mappings and symbols.

    If the qubits are provided as ints then the qubit will be named :code:`"q0"` and :code:`"q1"`, otherwise the qubits are assumed to be strings.

    Args:
        qcnn (hierarqcal.core.Qcnn): Qcnn object that describes the circuit architecture, consists of a sequence of motifs (hierarqcal.Qmotif)

    Returns:
        (tuple): Tuple containing:
            * circuit (qiskit.QuantumCircuit): QuantumCircuit object
            * symbols (tuple(Parameter)): Tuple of symbols (rotation angles) as a Qiskit Parameter object.
    """
    circuit = QuantumCircuit()
    new_bitnames = []
    for q in hierq.tail.Q:
        if type(q) == int:
            # If bits were provided as ints then the qubit will be named "q0" and "q1"
            circuit.add_register(QuantumRegister(1, f"q{q}"))
            new_bitnames.append(f"q{q}")
        else:
            # Assume bits are strings and in the correct QASM format
            circuit.add_register(QuantumRegister(1, q))
    if len(new_bitnames) > 0:
        hierq.update_Q(new_bitnames)
    if not (symbols is None):
        # If symbols were provided then set them
        hierq.set_symbols(symbols)
    else:
        if any([isinstance(symbol, sp.Symbol) for symbol in hierq.get_symbols()]):
            # If symbols are still symbolic, then convert to qiskit Parameter
            hierq.set_symbols([Parameter(symbol.name) if isinstance(symbol, sp.Symbol) else symbol for symbol in hierq.get_symbols()])

    for layer in hierq:
        # If layer is default mapping we need to set it to qiskit default
        if layer.is_default_mapping:
            qiskit_default_unitary = get_qiskit_default_unitary(layer)
            layer.set_edge_mapping(qiskit_default_unitary)
        for unitary in layer.edge_mapping:

            # If Qunitary.function is a string, we update the function to a Qiskit function
            # based on the provided string
            if isinstance(unitary.function, str):
                unitary = get_qiskit_circuit_from_instructions(unitary)

            circuit = unitary.function(
                bits=unitary.edge, symbols=unitary.symbols, 
                circuit=circuit, get_circuit_from_string=get_qiskit_circuit_from_instructions
            )
        if barriers and layer.next:
            # Add barrier between layers, except the last one.
            circuit.barrier()
    return circuit


def get_qiskit_circuit_from_instructions(qunitary):
    """ Takes `qunitary` whose `function` attribute is a string, converts the string into a function,
        and returns the updated `qunitary` class.

        Args: 
            `qunitary (hierarcqal.core.Qunitary)`
        Returns: 
            `qunitary (hierarcqal.core.Qunitary)`
        """

    # breaking down the qunitary.function string into a list of gate instructions
    instruction_list = qunitary.circuit_instructions

    # building a function from the list of gate instructions
    def circuit_fn(bits, symbols=None, circuit=None, **kwargs):
        qubits = [QuantumRegister(1, bits[i]) for i in range(qunitary.arity)]

        # for each gate in the instruction list
        s_indx = 0
        for circ_instruction in instruction_list:

            gate_name = circ_instruction.gate_name
            symbol_info = circ_instruction.symbol_info
            sub_bits = circ_instruction.sub_bits
            [n_symbols, param_ind, isinlist] = symbol_info

            # apply a gate 
            gate = getattr(circuit, gate_name)
            if n_symbols == 0:
                gate(*(qubits[i] for i in sub_bits))
            else:
                qub_list = (qubits[i] for i in sub_bits)
                if isinlist:
                    gate(*symbols[param_ind:param_ind+1], *qub_list)
                else:
                    sym_list = symbols[s_indx:s_indx+n_symbols]
                    gate(*sym_list,*qub_list)
            
            if not isinlist:
                s_indx += n_symbols
        return circuit
    
    qunitary.function = circuit_fn
    return qunitary