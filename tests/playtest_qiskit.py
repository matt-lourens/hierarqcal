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
    plot_motif,
    Qunitary,
)
from hierarqcal.qiskit.qiskit_circuits import V2, U2

# =======
# Breaks:
# =======
# N=10
# u = Qunitary("cnot()^10")
# qft = Qinit(N)  + Qpivot("1*1*1", stride=2,step=2, mapping=u, boundary="periodic")
# circuit= qft(backend="qiskit")
# circuit.draw("mpl")

# =======

# u = Qunitary("cnot()^10;cnot()^01")
# hierq = Qinit(10)+Qmask("1001", mapping=u)
# circuit = hierq(backend="qiskit")
# circuit.draw("mpl")

# # Motif to mask all qubits except outer two
# mask_middle = Qmask(global_pattern="0!0")
# unmask = Qunmask("previous")
# hrx_layer = mask_middle + Qcycle(mapping=Qunitary("RX(x)^0;H()^1")) + unmask
# hrx_layer_r = mask_middle + Qcycle(mapping=Qunitary("RX(x)^1;H()^0")) + unmask

# # Create ladder motif
# cnot_ladder = Qcycle(mapping=Qunitary("CNOT()^01"), boundary="open")
# cnot_ladder_r = Qcycle(mapping=Qunitary("CNOT()^01"), edge_order=[-1], boundary="open")
# rz_last = Qmask("*1", mapping = Qunitary("Rz(x)^0")) + Qunmask("previous") # TODO replace with Qpivot once it's ready
# ladder = cnot_ladder + rz_last + cnot_ladder_r

# # Create two excitations TODO term?
# excitation1 = hrx_layer + ladder + hrx_layer
# excitation2 = hrx_layer_r + ladder + hrx_layer_r

# hierq = Qinit(5) + excitation1 + excitation2
# hierq(backend="qiskit", barriers=False).draw("mpl")

# # Test masking
# u = Qunitary(V2, 0, 2)
# hierq = Qinit(8) + Qmask("!*", mapping=u)
# circuit = hierq(backend="qiskit")
# circuit.draw("mpl")

# u = Qunitary(V2, 0, 2)
# hierq = Qinit(8) + Qmask("*!*", mapping=u)
# circuit = hierq(backend="qiskit")
# circuit.draw("mpl")

# u = Qunitary(V2, 0, 2)
# hierq = Qinit(8) + Qmask("0!0", mapping=u)
# circuit = hierq(backend="qiskit")
# circuit.draw("mpl")


# # Test default
# hierq = Qinit(8) + Qcycle(1, 1, 0)
# circuit = hierq(backend="qiskit")
# circuit.draw("mpl")

# # Test masking
# u = Qunitary(V2, 0, 2)
# hierq = Qinit(8) + Qmask("1*1*1", mapping=u)
# circuit = hierq(backend="qiskit")
# circuit.draw("mpl")

# Test masking
# for stride in range(10):
#     u = Qunitary(V2, 0, 2)
#     hierq = Qinit(8) + Qmask("1*1*1",strides=stride, boundaries=["periodic","periodic","periodic"], mapping=u)
#     circuit = hierq(backend="qiskit")
#     circuit.draw("mpl")

# 'nearest tower'
u = Qunitary(V2, 0, 2)
hierq = Qinit(8) + Qmask("1*1*1", merge_between="1*1*1", mapping=u)
circuit = hierq(backend="qiskit")
circuit.draw("mpl")


# Test 'to much removal'
u = Qunitary("H()^0")
hierq = Qinit(9) + Qmask("1*1*1", mapping=u)*4
circuit = hierq(backend="qiskit")
circuit.draw("mpl")




# # Test masking
# hierq = Qinit(8) + (Qcycle(1, 1, 0) + Qmask("right")) * 3
# circuit = hierq(backend="qiskit")
# circuit.draw("mpl")

# # Test masking
# hierq = (
#     Qinit(8)
#     + (Qcycle(1, 1, 0) + Qmask("right")) * 3
#     + (Qunmask("previous") + Qcycle(1, 1, 0)) * 3
# )
# circuit = hierq(backend="qiskit")
# circuit.draw("mpl")

# # Test unmasking nothing
# hierq = Qinit(8) + Qunmask("previous") + Qunmask("previous")
# circuit = hierq(backend="qiskit")
# circuit.draw("mpl")

# Test unmasking right
# hierq = (
#     Qinit(8)
#     + (Qcycle(1, 1, 0) + Qmask("*!")) * 3
#     + (Qunmask("previous") + Qcycle(1, 1, 0)) * 4
# )
# circuit = hierq(backend="qiskit")
# circuit.draw("mpl")

# # Test unmasking patterns TODO test
# hierq = (
#     Qinit(8)
#     + (Qcycle(1, 1, 0) + Qmask("*!")) * 3
#     + Qunmask("!") + Qcycle(1, 1, 0)
# )
# circuit = hierq(backend="qiskit")
# circuit.draw("mpl")


# # Test qcnn
u = Qunitary("cnot()^01")
hierq = Qinit(8) + (Qcycle(1, 1, 0) + Qmask("right", mapping=u)) * 3
circuit = hierq(backend="qiskit")
circuit.draw("mpl")

# # Test sub TODO demo
u = Qunitary("crx(x)^01")
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