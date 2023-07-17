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

n = 10
random_int = np.random.randint(0, 2**n)
Target_string = bin(random_int)[2:].zfill(n)
print('With target',Target_string)

test = Qinit(n) + Qpivot(mapping=Qunitary("X()^0"), global_pattern=Target_string)
# create the circuit using the chose backend
circuit_test = test(backend="qiskit", barriers=True)#circuit.copy()
circuit_test.draw()




n = 4
qft = Qinit(n) + (
    Qpivot(global_pattern="1*", merge_within="*1", mapping=Qunitary("h()^0"))
    + Qpivot(mapping=Qunitary("cp(x)^01"), share_weights=False)
    + Qmask("1*")
) * (n)
circuit = qft(backend="qiskit")
circuit.draw("mpl")

N = 8
cnot = Qunitary("H()^0")
pivot_test = (
    Qinit(1)
    + (
        Qpivot(
            global_pattern="1*",
            merge_within="*1",
            merge_between=None,
            strides=[1, 1, 0],
            steps=[1, 1, 1],
            offsets=[0, 0, 0],
            boundaries=["open", "open", "periodic"],
            mapping=cnot,
        )
    )
    * 2
)
circuit = pivot_test(backend="qiskit")
circuit.draw("mpl")


N = 8
cnot = Qunitary("cnot()^01")
pivot_test = (
    Qinit(N)
    + (
        Qmask("1*")
        + Qpivot(
            global_pattern="1*",
            merge_within="*1",
            merge_between=None,
            strides=[1, 1, 0],
            steps=[1, 1, 1],
            offsets=[0, 0, 0],
            boundaries=["open", "open", "periodic"],
            mapping=cnot,
        )
    )
    * 2
)
circuit = pivot_test(backend="qiskit")
circuit.draw("mpl")


N = 8
cnot = Qunitary("cnot()^01")
pivot_test = Qinit(N) + Qpivot(
    global_pattern="1*",
    merge_within="*1",
    merge_between=None,
    strides=[1, 1, 0],
    steps=[1, 1, 1],
    offsets=[0, 0, 0],
    boundaries=["open", "open", "periodic"],
    mapping=cnot,
)
circuit = pivot_test(backend="qiskit")
circuit.draw("mpl")

ccnot = Qunitary("cnot()^03;cnot()^12")
pivot_test = Qinit(N) + Qpivot(
    global_pattern="*1",
    merge_within="*11",
    merge_between=None,
    strides=[1, 1, 0],
    steps=[1, 1, 1],
    offsets=[0, 0, 0],
    boundaries=["open", "open", "periodic"],
    mapping=ccnot,
)
circuit = pivot_test(backend="qiskit")
circuit.draw("mpl")

pivot_test = Qinit(N) + Qpivot(
    global_pattern="*1",
    merge_within="*11",
    merge_between=None,
    strides=[1, 1, 0],
    steps=[1, 2, 1],
    offsets=[0, 0, 0],
    boundaries=["open", "open", "periodic"],
    mapping=ccnot,
)
circuit = pivot_test(backend="qiskit")
circuit.draw("mpl")

t = Qunitary("cnot()^02;cnot()^12")
pivot_test = Qinit(N) + Qpivot(
    global_pattern="*1*",
    merge_within="*1",
    merge_between=None,
    strides=[1, 1, 0],
    steps=[1, 2, 1],
    offsets=[0, 0, 0],
    boundaries=["open", "open", "periodic"],
    mapping=t,
)
circuit = pivot_test(backend="qiskit")
circuit.draw("mpl")


pivot_test = Qinit(6) + Qpivot(
    global_pattern="1*1",
    merge_within="*1",
    merge_between=None,
    strides=[1, 1, 0],
    steps=[1, 1, 1],
    offsets=[0, 0, 0],
    boundaries=["open", "open", "periodic"],
    mapping=cnot,
)
circuit = pivot_test(backend="qiskit")
circuit.draw("mpl")

#######################################

u = Qunitary("cnot()^20;cnot()^21")

# qft = Qinit(N) + Qpivot("1*", local_pattern="11*", offset=0, mapping=u, boundary=boundary)
qft = Qinit(N) + Qpivot(
    global_pattern="1*",
    merge_within="11*",
    merge_between=None,
    strides=[1, 1, 0],
    steps=[1, 1, 1],
    offsets=[0, 0, 0],
    boundaries=["open", "open", "periodic"],
    mapping=u,
)
circuit = qft(backend="qiskit")
circuit.draw("mpl")


print("Hello")
# === break ===
# ========
# Test pivot
# QFT
# N = 3
# h_top = Qpivot(mapping=Qunitary("h()^0"))
# controlled_rs = Qpivot(mapping=Qunitary("crx(x)^01"), share_weights=False)
# qft = Qinit(N) + (h_top + controlled_rs + Qmask("1*")) * (N)
# circuit = qft(backend="qiskit")
# circuit.draw("mpl")

# # Base case 2-qubit uintary
# N = 10
# u = Qunitary("cnot()^01")

# # # open boundary:
# boundary = "open"
# pattern = "*1*1"
# # # default, offeset =0, step =1, stride =1

# qft = Qinit(N) + Qpivot(pattern, offset=0, mapping=u, boundary=boundary)
# circuit = qft(backend="qiskit")
# circuit.draw("mpl")

# # increase offset
# qft = Qinit(N) + Qpivot(pattern, offset=2, mapping=u, boundary=boundary)
# circuit = qft(backend="qiskit")
# circuit.draw("mpl")

# # increase step
# qft = Qinit(N) + Qpivot(pattern, step=2, mapping=u, boundary=boundary)
# circuit = qft(backend="qiskit")
# circuit.draw("mpl")

# # increase step and offset
# qft = Qinit(N) + Qpivot(pattern, offset=1, step=2, mapping=u, boundary=boundary)
# circuit = qft(backend="qiskit")
# circuit.draw("mpl")

# # periodic boundary:
# # # ordering seems off, even though the circuit is technically correct
# boundary = "periodic"
# pattern = "*1*1"

# # default, offeset =0, step =1, stride =1
# qft = Qinit(N) + Qpivot(pattern, offset=0, mapping=u, boundary=boundary)
# circuit = qft(backend="qiskit")
# circuit.draw("mpl")

# # increase offset

# qft = Qinit(N) + Qpivot(pattern, offset=2, mapping=u, boundary=boundary)
# circuit = qft(backend="qiskit")
# circuit.draw("mpl")

# # increase step
# qft = Qinit(N) + Qpivot(pattern, step=2, mapping=u, boundary=boundary)
# circuit = qft(backend="qiskit")
# circuit.draw("mpl")

# # increase step and offset
# qft = Qinit(N) + Qpivot(pattern, offset=1, step=2, mapping=u, boundary=boundary)
# circuit = qft(backend="qiskit")
# circuit.draw("mpl")


# # Complex pattern
# N = 10
# v = Qunitary("cnot()^01")
# qft = Qinit(N) + Qpivot(pattern="1*", stride=2, step=2, mapping=v, boundary="periodic")
# circuit = qft(backend="qiskit")
# circuit.draw("mpl")

# # 3 qubit unitary
# N = 9
# w = Qunitary("cnot()^01;cnot()^21")

# qft = Qinit(N) + Qpivot(pattern="1*", local_pattern="*1*", mapping=w, boundary=boundary)
# circuit = qft(backend="qiskit")
# circuit.draw("mpl")

# qft = Qinit(N) + Qpivot(pattern="1*", local_pattern="1*", mapping=w, boundary=boundary)
# circuit = qft(backend="qiskit")
# circuit.draw("mpl")

# qft = Qinit(N) + Qpivot(pattern="1*", local_pattern="1*1", mapping=w, boundary=boundary)
# circuit = qft(backend="qiskit")
# circuit.draw("mpl")
