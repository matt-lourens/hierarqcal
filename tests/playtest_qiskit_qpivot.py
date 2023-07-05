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


# Test pivot
# QFT
N = 3
h_top = Qpivot(mapping=Qunitary("h()^0"))
controlled_rs = Qpivot(mapping=Qunitary("crx(x)^01"), share_weights=False)
qft = Qinit(N) + (h_top + controlled_rs + Qmask("1*")) * (N)
circuit = qft(backend="qiskit")
circuit.draw("mpl")

# Base case 2-qubit uintary
N = 10
u = Qunitary("cnot()^01")

# open boundary:
boundary = "open"
pattern = "*1*1"
# default, offeset =0, step =1, stride =1

qft = Qinit(N) + Qpivot(pattern, offset=0, mapping=u, boundary=boundary)
circuit = qft(backend="qiskit")
circuit.draw("mpl")

# increase offset
qft = Qinit(N) + Qpivot(pattern, offset=2, mapping=u, boundary=boundary)
circuit = qft(backend="qiskit")
circuit.draw("mpl")

# increase step
qft = Qinit(N) + Qpivot(pattern, step=2, mapping=u, boundary=boundary)
circuit = qft(backend="qiskit")
circuit.draw("mpl")

# increase step and offset
qft = Qinit(N) + Qpivot(pattern, offset=1, step=2, mapping=u, boundary=boundary)
circuit = qft(backend="qiskit")
circuit.draw("mpl")

# periodic boundary:
# # ordering seems off, even though the circuit is technically correct
boundary = "periodic"
pattern = "*1*1"

# default, offeset =0, step =1, stride =1
qft = Qinit(N) + Qpivot(pattern, offset=0, mapping=u, boundary=boundary)
circuit = qft(backend="qiskit")
circuit.draw("mpl")

# increase offset

qft = Qinit(N) + Qpivot(pattern, offset=2, mapping=u, boundary=boundary)
circuit = qft(backend="qiskit")
circuit.draw("mpl")

# increase step
qft = Qinit(N) + Qpivot(pattern, step=2, mapping=u, boundary=boundary)
circuit = qft(backend="qiskit")
circuit.draw("mpl")

# increase step and offset
qft = Qinit(N) + Qpivot(pattern, offset=1, step=2, mapping=u, boundary=boundary)
circuit = qft(backend="qiskit")
circuit.draw("mpl")


# Complex pattern
N = 10
v = Qunitary("cnot()^01")
qft = Qinit(N) + Qpivot(pattern="1*", stride=2, step=2, mapping=v, boundary="periodic")
circuit = qft(backend="qiskit")
circuit.draw("mpl")

# 3 qubit unitary
N = 9
w = Qunitary("cnot()^01;cnot()^21")

qft = Qinit(N) + Qpivot(pattern="1*", local_pattern="*1*", mapping=w, boundary=boundary)
circuit = qft(backend="qiskit")
circuit.draw("mpl")

qft = Qinit(N) + Qpivot(pattern="1*", local_pattern="1*", mapping=w, boundary=boundary)
circuit = qft(backend="qiskit")
circuit.draw("mpl")

qft = Qinit(N) + Qpivot(pattern="1*", local_pattern="1*1", mapping=w, boundary=boundary)
circuit = qft(backend="qiskit")
circuit.draw("mpl")

print("Hello")
