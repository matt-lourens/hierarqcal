"""
TODO one day these will turn into nice tests, the idea is to run them one by one in debug mode. There are a lot of plots so make sure you have break points
"""
import numpy as np
from hierarqcal import (
    Qhierarchy,
    Qcycle,
    Qpivot,
    Qpermute,
    Qmask,
    Qunmask,
    Qinit,
    Qmotif,
    Qmotifs,
    # plot_motifs,
    # plot_motif,
    Qunitary,
)
from hierarqcal.qiskit.qiskit_circuits import V2, U2



# Test default
hierq = Qinit(8) + Qcycle(1, 1, 0)
circuit = hierq(backend="qiskit")
circuit.draw("mpl")

# Test masking
u = Qunitary(V2, 0, 2)
hierq = Qinit(8) + Qmask("1*1*1", mapping=u)
circuit = hierq(backend="qiskit")
circuit.draw("mpl")

# Test masking
u = Qunitary(V2, 0, 2)
hierq = Qinit(8) + Qmask("1*1*1", connection_type="nearest_tower", mapping=u)
circuit = hierq(backend="qiskit")
circuit.draw("mpl")

# Test masking TODO DEMO
u = Qunitary(V2, 0, 2)
hierq = Qinit(8) + Qmask("1**1", connection_type="nearest_tower", mapping=u)*4
circuit = hierq(backend="qiskit")
circuit.draw("mpl")

# Test masking
hierq = Qinit(8) + (Qcycle(1, 1, 0) + Qmask("right")) * 3
circuit = hierq(backend="qiskit")
circuit.draw("mpl")

# Test masking
hierq = (
    Qinit(8)
    + (Qcycle(1, 1, 0) + Qmask("right")) * 3
    + (Qunmask("previous") + Qcycle(1, 1, 0)) * 3
)
circuit = hierq(backend="qiskit")
circuit.draw("mpl")

# Test unmasking nothing
hierq = Qinit(8) + Qunmask("previous") + Qunmask("previous")
circuit = hierq(backend="qiskit")
circuit.draw("mpl")

# Test unmasking nothing
hierq = (
    Qinit(8)
    + (Qcycle(1, 1, 0) + Qmask("right")) * 3
    + (Qunmask("previous") + Qcycle(1, 1, 0)) * 4
)
circuit = hierq(backend="qiskit")
circuit.draw("mpl")

# Test unmasking patterns
hierq = (
    Qinit(8)
    + (Qcycle(1, 1, 0) + Qmask("right")) * 3
    + (Qunmask("right") + Qcycle(1, 1, 0)) * 4
)
circuit = hierq(backend="qiskit")
circuit.draw("mpl")


# # Test qcnn
u = Qunitary(V2, 0, 2)
hierq = Qinit(8) + (Qcycle(1, 1, 0) + Qmask("right", mapping=u)) * 3
circuit = hierq(backend="qiskit")
circuit.draw("mpl")

# # Test sub TODO demo
u = Qunitary(V2, 1, 2)
sub = Qinit(3) + Qpermute(mapping=u)
hierq = Qinit(15) + Qcycle(
    1, len(sub.tail.Q), 0, mapping=sub, share_weights=True, boundary="open"
)
circuit = hierq(backend="qiskit")
circuit.draw("mpl")

# Test sub sub TODO demo
u = Qunitary(U2, 1, 2)
subsub = Qinit(3) + Qpermute(mapping=u)
sub = Qinit(12) + Qcycle(1, 2, 0, mapping=subsub, share_weights=True, boundary="open")
hierq = Qinit(24) + Qcycle(1, 11, 0, mapping=sub, share_weights=True, boundary="open")
circuit = hierq(backend="qiskit")
circuit.draw("mpl")


# # Test sub weights TODO demo
u = Qunitary(U2, 1, 2)
sub = Qinit(3) + Qpermute(mapping=u, share_weights=False)
hierq = Qinit(15) + Qcycle(
    1, len(sub.tail.Q), 0, mapping=sub, share_weights=False, boundary="open"
)
circuit = hierq(backend="qiskit")
circuit.draw("mpl")

u = Qunitary(U2, 1, 2)
sub = Qinit(3) + Qpermute(mapping=u, share_weights=False)
hierq = Qinit(15) + Qcycle(
    1, len(sub.tail.Q), 0, mapping=sub, share_weights=False, boundary="open"
)
circuit = hierq(backend="qiskit")
circuit.draw("mpl")
u = Qunitary(U2, 1, 2)
sub = Qinit(3) + Qpermute(mapping=u, share_weights=False)
hierq = Qinit(15) + Qcycle(
    1, len(sub.tail.Q), 0, mapping=sub, share_weights=True, boundary="open"
)
circuit = hierq(backend="qiskit")
circuit.draw("mpl")
u = Qunitary(U2, 1, 2)
sub = Qinit(3) + Qpermute(mapping=u, share_weights=True)
hierq = Qinit(15) + Qcycle(
    1, len(sub.tail.Q), 0, mapping=sub, share_weights=False, boundary="open"
)
circuit = hierq(backend="qiskit")
circuit.draw("mpl")
u = Qunitary(U2, 1, 2)
sub = Qinit(3) + Qpermute(mapping=u, share_weights=True)
hierq = Qinit(15) + Qcycle(
    1, len(sub.tail.Q), 0, mapping=sub, share_weights=True, boundary="open"
)
circuit = hierq(backend="qiskit")
circuit.draw("mpl")


# Test set symbols
u = Qunitary(U2, 1, 2)
v = Qunitary(V2, 0, 2)
m1 = Qinit(4) + (Qcycle(mapping=u, share_weights=True) + Qmask("right", mapping=v)) * 2
hierq = Qinit(10) + Qcycle(1, 3, mapping=m1, boundary="open", share_weights=True)
hierq.set_symbols(np.random.uniform(0, 2 * np.pi, hierq.n_symbols))
list(hierq.get_symbols())
circuit = hierq(backend="qiskit")
circuit.draw("mpl")



# Test pivot
N = 8
h_top = Qpivot(mapping=Qunitary("h()^0"))
controlled_rs = Qpivot(mapping=Qunitary("crx(x)^01"), share_weights=False)
qft = Qinit(N)  + (h_top + controlled_rs + Qmask("1*"))*(N)
circuit= qft(backend="qiskit")
circuit.draw("mpl")

print("Hello")