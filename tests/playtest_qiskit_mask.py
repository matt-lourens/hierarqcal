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

# Strides between, open between
# N = 8
# for stride in range(2 * N):
#     u = Qunitary("cnot()^01")
#     hierq = (
#         Qinit(N)
#         + Qmask(
#             "1*1",
#             strides=[1, 1, stride],
#             boundaries=["open", "open", "open"],
#             merge_pattern="01",
#             mapping=u,
#         )
#         + Qcycle(mapping=Qunitary("h()^0"))
#     )
#     circuit = hierq(backend="qiskit")
#     circuit.draw("mpl")

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


N = 9
for stride in range(2 * N):
    u = Qunitary("cnot()^01;cnot()^21")
    hierq = (
        Qinit(N)
        + Qmask(
            "101",
            strides=[1, 1, stride],
            steps=[1, 1, 2],
            boundaries=["open", "open", "periodic"],
            merge_within="101",
            merge_between=None,
            mapping=u,
        )
        + Qcycle(mapping=Qunitary("h()^0"))
    )
    circuit = hierq(backend="qiskit")
    circuit.draw("mpl")