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

# N = 10
# u = Qunitary("H()^0")
# qft = Qinit(N) + Qpivot(
#     global_pattern="1*1*",
#     merge_within="*1",
#     merge_between=None,
#     strides=[1, 1, 0],
#     steps=[1, 1, 1],
#     offsets=[0, 0, 0],
#     boundaries=["open", "open", "periodic"],
#     mapping=u, )
# circuit = qft(backend="qiskit")
# circuit.draw("mpl")


# u = Qunitary("cnot()^01")
# qft = Qinit(N) + Qpivot(
#     global_pattern="1*",
#     merge_within="*1",
#     merge_between=None,
#     strides=[1, 1, 0],
#     steps=[1, 1, 1],
#     offsets=[0, 0, 0],
#     boundaries=["open", "open", "periodic"],
#     mapping=u, )
# circuit = qft(backend="qiskit")
# circuit.draw("mpl")


# u = Qunitary("cnot()^01")
# qft = Qinit(N) + Qpivot(
#     global_pattern="*1*",
#     merge_within="*1",
#     merge_between=None,
#     strides=[1, 1, 0],
#     steps=[1, 1, 1],
#     offsets=[0, 0, 0],
#     boundaries=["open", "open", "periodic"],
#     mapping=u, )
# circuit = qft(backend="qiskit")
# circuit.draw("mpl")


# u = Qunitary("cnot()^01")
# qft = Qinit(N) + Qpivot(
#     global_pattern="*1*1",
#     merge_within="*1",
#     merge_between=None,
#     strides=[1, 1, 0],
#     steps=[1, 1, 1],
#     offsets=[0, 0, 0],
#     boundaries=["open", "open", "periodic"],
#     mapping=u, )
# circuit = qft(backend="qiskit")
# circuit.draw("mpl")

# u = Qunitary("cnot()^01")
# qft = Qinit(N) + Qpivot(
#     global_pattern="*11*",
#     merge_within="*1",
#     merge_between=None,
#     strides=[1, 1, 0],
#     steps=[1, 1, 1],
#     offsets=[0, 0, 0],
#     boundaries=["open", "open", "periodic"],
#     mapping=u, )
# circuit = qft(backend="qiskit")
# circuit.draw("mpl")


# u = Qunitary("cnot()^01")
# qft = Qinit(N) + Qpivot(
#     global_pattern="*11*",
#     merge_within="*1",
#     merge_between=None,
#     strides=[1, 1, 0],
#     steps=[1, 1, 1],
#     offsets=[0, 0, 0],
#     boundaries=["open", "open", "periodic"],
#     mapping=u, )
# circuit = qft(backend="qiskit")
# circuit.draw("mpl")
# 
# u = Qunitary("cnot()^01;cnot()^02")
# qft = Qinit(N) + Qpivot(
#     global_pattern="*1*",
#     merge_within="*11",
#     merge_between=None,
#     strides=[1, 1, 0],
#     steps=[1, 1, 1],
#     offsets=[0, 0, 0],
#     boundaries=["open", "open", "periodic"],
#     mapping=u, )
# circuit = qft(backend="qiskit")
# circuit.draw("mpl")


n= 5
k =1
symbol_fn = lambda x, ns, ne: np.pi*2**(-ne)
h = Qunitary("h()^0")
c_p = Qunitary("cp(x)^01")
U_k = Qinit(n) + Qmask("1"*(k-1)+"*") + Qpivot(mapping=h) + Qpivot(mapping=c_p, share_weights=False, symbol_fn = symbol_fn)
circuit = U_k(backend="qiskit")
circuit.draw("mpl")

print('Hello')
