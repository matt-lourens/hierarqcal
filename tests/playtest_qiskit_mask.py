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
from hierarqcal.qiskit.qiskit_circuits import V2, U2, U3

# Strides between, open between
N = 8
for stride in range(2 * N):
    u = Qunitary("cnot()^01")
    hierq = (
        Qinit(N)
        + Qmask(
            "1*1",
            strides=[1, 1, stride],
            boundaries=["open", "open", "open"],
            merge_pattern="01",
            mapping=u,
        )
        + Qcycle(mapping=Qunitary("h()^0"))
    )
    circuit = hierq(backend="qiskit")
    circuit.draw("mpl")

# # Strides between, periodic between
# N = 8
# for stride in range(2 * N):
#     u = Qunitary("cnot()^01")
#     hierq = (
#         Qinit(N)
#         + Qmask(
#             "1*1",
#             strides=[1, 1, stride],
#             boundaries=["open", "open", "periodic"],
#             merge_pattern="01",
#             mapping=u,
#         )
#         + Qcycle(mapping=Qunitary("h()^0"))
#     )
#     circuit = hierq(backend="qiskit")
#     circuit.draw("mpl")

# # merge between
# N = 8
# for stride in range(2 * N):
#     u = Qunitary("cnot()^01")
#     hierq = (
#         Qinit(N)
#         + Qmask(
#             "1*1",
#             strides=[1, 1, stride],
#             steps=[1, 1, 1],
#             boundaries=["open", "open", "periodic"],
#             merge_within="01",
#             merge_between="1*1",
#             mapping=u,
#         )
#         + Qcycle(mapping=Qunitary("h()^0"))
#     )
#     circuit = hierq(backend="qiskit")
#     circuit.draw("mpl")


# N = 8
# for stride in range(2 * N):
#     u = Qunitary("cnot()^01")
#     hierq = (
#         Qinit(N)
#         + Qmask(
#             "1*1",
#             strides=[1, 1, stride],
#             steps=[1, 1, 1],
#             boundaries=["open", "open", "periodic"],
#             merge_within="01",
#             merge_between="*1*",
#             mapping=u,
#         )
#         + Qcycle(mapping=Qunitary("h()^0"))
#     )
#     circuit = hierq(backend="qiskit")
#     circuit.draw("mpl")

# N = 9
# for stride in range(2 * N):
#     u = Qunitary("cnot()^01;cnot()^21")
#     hierq = (
#         Qinit(N)
#         + Qmask(
#             "101",
#             strides=[1, 1, stride],
#             steps=[1, 1, 2],
#             boundaries=["open", "open", "periodic"],
#             merge_within="101",
#             merge_between=None,
#             mapping=u,
#         )
#         + Qcycle(mapping=Qunitary("h()^0"))
#     )
#     circuit = hierq(backend="qiskit")
#     circuit.draw("mpl")


# N = 9
# for stride in range(2 * N):
#     u = Qunitary("cnot()^01;cnot()^21")
#     hierq = (
#         Qinit(N)
#         + Qmask(
#             "101",
#             strides=[1, 1, stride],
#             steps=[1, 1, 2],
#             boundaries=["open", "open", "periodic"],
#             merge_within="101",
#             merge_between="*1*1*1*",
#             mapping=u,
#         )
#         + Qcycle(mapping=Qunitary("h()^0"))
#     )
#     circuit = hierq(backend="qiskit")
#     circuit.draw("mpl")


# N = 9
# u = Qunitary("cnot()^01;cnot()^21")
# #u = Qunitary(function=U3, arity=3, n_symbols=2)
# hierq = (
#     Qinit(N)
#     + Qmask(
#         "101",
#         strides=[1, 1, 0],
#         steps=[2, 1, 1],
#         boundaries=["open", "open", "open"],
#         merge_within="101",
#         merge_between=None,
#         mapping=u,
#     )
#     + Qcycle(mapping=Qunitary("h()^0"))
# )
# circuit = hierq(backend="qiskit")
# circuit.draw("mpl")

# Right
# N = 8
# u = Qunitary("cnot()^01")
# #u = Qunitary(function=U3, arity=3, n_symbols=2)
# hierq = (
#     Qinit(N)
#     + (Qmask(
#         "*!",
#         strides=[1, 1, 0],
#         steps=[1, 1, 1],
#         boundaries=["open", "open", "open"],
#         merge_within="10",
#         merge_between=None,
#         mapping=u,
#     )
#     + Qcycle(mapping=Qunitary("h()^0")))*3
# )
# circuit = hierq(backend="qiskit")
# circuit.draw("mpl")

# Left
# N = 8
# u = Qunitary("cnot()^01")
# #u = Qunitary(function=U3, arity=3, n_symbols=2)
# hierq = (
#     Qinit(N)
#     + (Qmask(
#         "*!",
#         strides=[1, 1, 0],
#         steps=[1, 1, 1],
#         boundaries=["open", "open", "open"],
#         merge_within="10",
#         merge_between=None,
#         mapping=u,
#     )
#     + Qcycle(mapping=Qunitary("h()^0")))*3
# )
# circuit = hierq(backend="qiskit")
# circuit.draw("mpl")

# Outside in
# N = 8
# u = Qunitary("cnot()^01")
# #u = Qunitary(function=U3, arity=3, n_symbols=2)
# hierq = (
#     Qinit(N)
#     + (Qmask(
#         "!*!",
#         strides=[1, 1, 0],
#         steps=[1, 1, 1],
#         boundaries=["open", "open", "open"],
#         merge_within="10",
#         merge_between=None,
#         mapping=u,
#     )
#     + Qcycle(mapping=Qunitary("h()^0")))*3
# )
# circuit = hierq(backend="qiskit")
# circuit.draw("mpl")

# inside out
# N = 8
# u = Qunitary("cnot()^01")
# #u = Qunitary(function=U3, arity=3, n_symbols=2)
# hierq = (
#     Qinit(N)
#     + (Qmask(
#         "*!*",
#         strides=[1, 1, 0],
#         steps=[1, 1, 1],
#         boundaries=["open", "open", "open"],
#         merge_within="10",
#         merge_between=None,
#         mapping=u,
#     )
#     + Qcycle(mapping=Qunitary("h()^0")))*3
# )
# circuit = hierq(backend="qiskit")
# circuit.draw("mpl")

# even
# N = 8
# u = Qunitary("cnot()^01")
# #u = Qunitary(function=U3, arity=3, n_symbols=2)
# hierq = (
#     Qinit(N)
#     + (Qmask(
#         "01",
#         strides=[1, 1, 0],
#         steps=[1, 1, 1],
#         boundaries=["open", "open", "open"],
#         merge_within="10",
#         merge_between=None,
#         mapping=u,
#     )
#     + Qcycle(mapping=Qunitary("h()^0")))*3
# )
# circuit = hierq(backend="qiskit")
# circuit.draw("mpl")

# # odd
# N = 8
# u = Qunitary("cnot()^01")
# #u = Qunitary(function=U3, arity=3, n_symbols=2)
# hierq = (
#     Qinit(N)
#     + (Qmask(
#         "10",
#         strides=[1, 1, 0],
#         steps=[1, 1, 1],
#         boundaries=["open", "open", "open"],
#         merge_within="10",
#         merge_between=None,
#         mapping=u,
#     )
#     + Qcycle(mapping=Qunitary("h()^0")))*3
# )
# circuit = hierq(backend="qiskit")
# circuit.draw("mpl")

# MERA
# N=16
# c1 = Qcycle(1,2,1,mapping=Qunitary("crx(x)^01"), boundary="open")
# c2 = Qcycle(1,2,0,mapping=Qunitary("crx(x)^01"), boundary="open")
# m1 = Qmask("1001", merge_within="1001", steps=[2,2,1], boundaries=["open","open","open"] ,mapping=Qunitary("cnot()^01;cnot()^32"))
# hierq = Qinit(N) + c1 + (m1+c2)*int(np.log2(N))
# circuit = hierq(backend="qiskit")
# circuit.draw("mpl")




print("hi")