import itertools as it
from hierarqcal import (
    Qcycle,
    Qmotif,
    Qinit,
    Qmask,
    Qpermute,
    Qpivot,
    plot_circuit,
    plot_motif,
    get_tensor_as_f,
    Qunitary,
)
from functools import reduce

import cirq
from cirq.contrib.svg import SVGCircuit


def U1(bits, symbols=None, circuit=None):
    q0, q1 = cirq.LineQubit(bits[0]), cirq.LineQubit(bits[1])
    circuit += cirq.rx(symbols[0]).on(q1).controlled_by(q0)
    return circuit


def U2(bits, symbols=None, circuit=None):
    q0, q1 = cirq.LineQubit(bits[0]), cirq.LineQubit(bits[1])
    circuit += cirq.ry(symbols[0]).on(q1).controlled_by(q0)
    return circuit


def U3(bits, symbols=None, circuit=None):
    q0, q1 = cirq.LineQubit(bits[0]), cirq.LineQubit(bits[1])
    circuit += cirq.rz(symbols[0]).on(q1).controlled_by(q0)
    return circuit


u1 = Qunitary(U1, n_symbols=1, arity=2)
u2 = Qunitary(U2, n_symbols=1, arity=2)
u3 = Qunitary(U3, n_symbols=1, arity=2)

nq = 4
qubit_indices = [i for i in range(nq)]
n_gates = 3
gates = [u1, u2, u3]
depth = 3

gates_idx = {i for i in range(n_gates)}
subsets = list(it.product(gates_idx, repeat=depth))
all_possibilites = dict()
for subset in subsets:
    if len(subset) > 0:
        # Get the different orderings ex gate1+gate2 is different from gate2+gate1
        orderings = set(it.permutations(subset))
        gate_permutations = dict()
        for gate in subset:
            gate_permutations[gate] = list(it.permutations([i for i in range(nq)], 2))
        combined_2 = dict()
        for ordering in orderings:
            combined_2[ordering] = list(
                it.product(*[gate_permutations[gate] for gate in ordering])
            )
        all_possibilites[subset] = combined_2

for subset, orderings in all_possibilites.items():
    for ordering, permutations in orderings.items():
        for permutation in permutations:
            circuit_motif = [
                Qmotif(E=[permutation], mapping=gates[gate_ind])
                for permutation, gate_ind in zip(permutation, ordering)
            ]
            hierq = Qinit(qubit_indices) + reduce(lambda a, b: a + b, circuit_motif)
            circuit = hierq(backend="cirq")
            print(circuit)
