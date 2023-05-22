"""
TODO one day these will turn into nice tests, the idea is to run them one by one in debug mode. There are a lot of plots so make sure you have break points
"""
import numpy as np
from hierarqcal import (
    Qhierarchy,
    Qcycle,
    Qpermute,
    Qmask,
    Qunmask,
    Qinit,
    Qmotif,
    Qmotifs,
    plot_motifs,
    plot_motif,
    Qunitary,
)
import sympy as sp
import pennylane as qml


def get_circuit(hierq):
    dev = qml.device("default.qubit", wires=hierq.tail.Q)

    @qml.qnode(dev)
    def circuit():
        if isinstance(next(hierq.get_symbols(), False), sp.Symbol):
            # Pennylane doesn't support symbolic parameters, so if no symbols were set (i.e. they are still symbolic), we initialize them randomly
            hierq.set_symbols(np.random.uniform(0, 2 * np.pi, hierq.n_symbols))
        hierq(backend="pennylane")  # This executes the compute graph in order
        return [qml.expval(qml.PauliZ(wire)) for wire in hierq.tail.Q]

    return circuit


def draw_circuit(circuit, **kwargs):
    fig, ax = qml.draw_mpl(circuit)(**kwargs)


# Define unitaries
def V1(bits, symbols=None):
    qml.CNOT(wires=[bits[0], bits[1]])


def U1(bits, symbols=None):
    qml.CRX(symbols[0], wires=[bits[0], bits[1]])


u1 = Qunitary(U1, 2, 2)
# Create architecture
# Test == wrong symbol allocation
motif = Qinit(8) + Qcycle(1, 1, 0, mapping=u1)
# motif[1].set_symbols([np.pi / 2])

# Test == correct symbol counts
motif = (
    Qinit(8) + Qcycle(1, 1, 0, mapping=u1, share_weights=False) + Qcycle(1, mapping=u1)
)
list(motif.get_symbols())


# Test == empty symbol counts
def U(bits, symbols=None):
    qml.CRZ(symbols[0], wires=[bits[0], bits[1]])
    qml.CRZ(symbols[1], wires=[bits[0], bits[1]])


u = Qunitary(U, 2, 2)
hierq = (
    Qinit(8) + Qcycle(1, 1, 0, mapping=u, share_weights=False) + Qcycle(1, mapping=u)
)
circuit = get_circuit(hierq)
draw_circuit(circuit)
list(hierq.get_symbols())
list(hierq[0].get_symbols())
list(hierq[1].get_symbols())
list(hierq[2].get_symbols())
hierq[2].edge_mapping[0].symbols


hierq[2].set_symbols([np.pi / 4, np.pi / 2])
circuit = get_circuit(hierq)
print(circuit())


# # == subhierarchy ==
def U(bits, symbols=None):
    qml.CRY(symbols[0], wires=[bits[0], bits[1]])


u = Qunitary(U, 1, 2)
sub = Qinit(3) + Qpermute(mapping=u)
motif = Qinit(15) + Qcycle(
    1, len(sub.tail.Q), 0, mapping=sub, share_weights=True, boundary="open"
)
circuit = get_circuit(motif)
draw_circuit(circuit)
# == defaults ==

motif = Qinit(9) + Qcycle(1)
circuit = get_circuit(motif)
draw_circuit(circuit)

# == Mask ==

motif = Qinit(8) + Qcycle(1) + Qmask() + Qcycle(1)
circuit = get_circuit(motif)
draw_circuit(circuit)

# == Qinit ==

motif = Qinit([7, 8, 9, 4]) + Qcycle(1) + Qmask(pattern="right") + Qcycle()
circuit = get_circuit(motif)
draw_circuit(circuit)

# Subs
u = Qunitary(U, 1, 2)
v = Qunitary(V1, 0, 2)
sub = (
    Qinit(8)
    + Qcycle(1, mapping=u)
    + Qmask(pattern="right", mapping=v)
    + Qcycle(mapping=u)
)
motif = Qinit(16) + Qcycle(mapping=sub, boundary="open")
circuit = get_circuit(motif)
draw_circuit(circuit)

# sub sub
u = Qunitary(U1, 2, 2)
subsub = Qinit(3) + Qpermute(mapping=u)
sub = Qinit(12) + Qcycle(1, 2, 0, mapping=subsub, share_weights=True, boundary="open")
hierq = Qinit(24) + Qcycle(1, 11, 0, mapping=sub, share_weights=True, boundary="open")
circuit = get_circuit(motif)
draw_circuit(circuit)


# == Qmask arity>2 ==
def U3(bits, symbols=None):
    qml.CRY(symbols[0], wires=[bits[0], bits[1]])
    qml.CRY(symbols[1], wires=[bits[2], bits[1]])


def U2(bits, symbols=None):
    qml.CRY(symbols[0], wires=[bits[0], bits[1]])


u3 = Qunitary(U3, 2, 3)
u2 = Qunitary(U2, 2, 2)
motif = Qinit(8) + Qmask("101", stride=0, step=1, offset=2, mapping=u3)
circuit = get_circuit(motif)
draw_circuit(circuit)


# == Qmask sub==
def U3(bits, symbols=None):
    qml.CRY(symbols[0], wires=[bits[0], bits[1]])
    qml.CRY(symbols[1], wires=[bits[2], bits[1]])


def U2(bits, symbols=None):
    qml.CRY(symbols[0], wires=[bits[0], bits[1]])


u3 = Qunitary(U3, 2, 3)
u2 = Qunitary(U2, 2, 2)
sub = Qinit(4) + Qmask(pattern="right", mapping=u2)
motif = Qinit(16) + Qcycle(0, mapping=sub)
circuit = get_circuit(motif)
draw_circuit(circuit)


# == Qmask sub==
def U3(bits, symbols=None):
    qml.CRY(symbols[0], wires=[bits[0], bits[1]])
    qml.CRY(symbols[1], wires=[bits[2], bits[1]])


def U2(bits, symbols=None):
    qml.CRY(symbols[0], wires=[bits[0], bits[1]])


u3 = Qunitary(U3, 2, 3)
u2 = Qunitary(U2, 2, 2)
sub = Qinit(4) + Qmask(pattern="right", mapping=u2)
motif = (
    Qinit(16)
    + Qmask(stride=0, step=4, pattern="0011", mapping=sub, boundary="open")
    + Qcycle()
)
circuit = get_circuit(motif)
draw_circuit(circuit)


# == Qmask with/without==
def U3(bits, symbols=None):
    qml.CRY(symbols[0], wires=[bits[0], bits[1]])
    qml.CRY(symbols[1], wires=[bits[2], bits[1]])


def U2(bits, symbols=None):
    qml.CRY(symbols[0], wires=[bits[0], bits[1]])


u3 = Qunitary(U3, 2, 3)
u2 = Qunitary(U2, 1, 2)
sub = Qinit(4) + Qmask(pattern="right")
motif = (
    Qinit(16)
    + Qmask(stride=0, step=4, pattern="0011", mapping=sub, boundary="open")
    + Qcycle()
)
# === Pennylane ===
circuit = get_circuit(motif)
draw_circuit(circuit)


# == Qmask with/without==
def U3(bits, symbols=None):
    qml.CRY(symbols[0], wires=[bits[0], bits[1]])
    qml.CRY(symbols[1], wires=[bits[2], bits[1]])


def U2(bits, symbols=None):
    qml.CRY(symbols[0], wires=[bits[0], bits[1]])


u3 = Qunitary(U3, 2, 3)
u2 = Qunitary(U2, 2, 2)
motif1 = Qinit(8) + Qcycle() + Qmask(pattern="right") + Qcycle()
motif2 = Qinit(8) + Qcycle() + Qmask(pattern="01") + Qcycle()
motif3 = Qinit(8) + Qcycle() + Qmask(pattern="1*") + Qcycle()
motif4 = Qinit(8) + Qcycle() + Qmask(pattern="11*") + Qcycle()
motif5 = Qinit(8) + Qcycle() + Qmask(pattern="11*11") + Qcycle()
motif6 = Qinit(4) + Qcycle() + Qmask(pattern="11*11") + Qcycle()
motif7 = Qinit(3) + Qcycle() + Qmask(pattern="11*11") + Qcycle()
motif8 = Qinit(8) + (Qcycle() + Qmask(pattern="1*1")) * 6
# === Pennylane ===
circuit = get_circuit(motif1)
draw_circuit(circuit)
circuit = get_circuit(motif2)
draw_circuit(circuit)
circuit = get_circuit(motif3)
draw_circuit(circuit)
circuit = get_circuit(motif4)
draw_circuit(circuit)
circuit = get_circuit(motif5)
draw_circuit(circuit)
circuit = get_circuit(motif6)
draw_circuit(circuit)
circuit = get_circuit(motif7)
draw_circuit(circuit)
circuit = get_circuit(motif8)
draw_circuit(circuit)


# == Qunmask with/without==
def U3(bits, symbols=None):
    qml.CRY(symbols[0], wires=[bits[0], bits[1]])
    qml.CRY(symbols[1], wires=[bits[2], bits[1]])


def U2(bits, symbols=None):
    qml.CRY(symbols[0], wires=[bits[0], bits[1]])


u3 = Qunitary(U3, 2, 3)
u2 = Qunitary(U2, 2, 2)
motif1 = (
    Qinit(8)
    + Qcycle()
    + Qmask(pattern="right")
    + Qcycle()
    + Qunmask(pattern="right")
    + Qcycle()
)
circuit = get_circuit(motif8)
draw_circuit(circuit)


# == 1 qubit unitaries==
def U1(bits, symbols=None):
    qml.Hadamard(wires=[bits[0]])


def U2(bits, symbols=None):
    qml.RX(symbols[0], wires=[bits[0]])


u1 = Qunitary(U1, 0, 1)
u2 = Qunitary(U2, 1, 1)
motif1 = Qinit(8) + Qcycle(mapping=u1)
motif2 = Qinit(8) + Qcycle(mapping=u1) * 2
motif3 = Qinit(8) + Qcycle(mapping=u2) * 2
motif4 = Qinit(8) + Qcycle(1, 2, 0, mapping=u2) * 2
motif5 = Qinit(8) + Qcycle(1, 2, 2, mapping=u2) * 2
circuit = get_circuit(motif1)
draw_circuit(circuit)
circuit = get_circuit(motif2)
draw_circuit(circuit)
circuit = get_circuit(motif3)
draw_circuit(circuit)
circuit = get_circuit(motif4)
draw_circuit(circuit)
circuit = get_circuit(motif5)
draw_circuit(circuit)


# == Permute boundary and arity  ==
def U1(bits, symbols=None):
    qml.Hadamard(wires=[bits[0]])


def U2(bits, symbols=None):
    qml.RX(symbols[0], wires=[bits[0]])


def U3(bits, symbols=None):
    qml.CRY(symbols[0], wires=[bits[0], bits[1]])


def U4(bits, symbols=None):
    qml.CRY(symbols[0], wires=[bits[0], bits[1]])
    qml.CRY(symbols[1], wires=[bits[2], bits[1]])


u1 = Qunitary(U1, 0, 1)
u2 = Qunitary(U2, 1, 1)
u3 = Qunitary(U3, 1, 2)
u4 = Qunitary(U4, 2, 3)
motif1 = Qinit(8) + Qpermute(mapping=u3)
motif2 = Qinit(8) + Qpermute(mapping=u4)
motif3 = Qinit(8) + Qpermute(mapping=u2)
motif4 = Qinit(8) + Qcycle(1, 2, 0, mapping=u2) * 2
motif5 = Qinit(8) + Qcycle(1, 2, 2, mapping=u2) * 2
circuit = get_circuit(motif1)
draw_circuit(circuit)
circuit = get_circuit(motif2)
draw_circuit(circuit)
circuit = get_circuit(motif3)
draw_circuit(circuit)
circuit = get_circuit(motif4)
draw_circuit(circuit)
circuit = get_circuit(motif5)
draw_circuit(circuit)
