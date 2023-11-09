from hierarqcal import (
    Qcycle,
    Qmotif,
    Qinit,
    Qmask,
    Qunmask,
    Qpermute,
    Qpivot,
    plot_circuit,
    plot_motif,
    get_tensor_as_f,
    Qunitary,
)
import numpy as np
import sympy as sp
import itertools as it

# ====== Matrices
toffoli_m = np.identity(8)
toffoli_m[6, 6] = 0
toffoli_m[6, 7] = 1
toffoli_m[7, 6] = 1
toffoli_m[7, 7] = 0
cnot_m = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
h_m = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])

# ====== Gates level 0
toffoli = Qunitary(get_tensor_as_f(toffoli_m), 0, 3)
cnot = Qunitary(get_tensor_as_f(cnot_m), 0, 2)
hadamard = Qunitary(get_tensor_as_f(h_m), 0, 1)

# ====== Motifs level 1
carry = (
    Qinit(4)
    + Qmotif(E=[(1, 2, 3)], mapping=toffoli)
    + Qmotif(E=[(1, 2)], mapping=cnot)
    + Qmotif(E=[(0, 2, 3)], mapping=toffoli)
)
sum = Qinit(3) + Qpivot("*1", merge_within="01", mapping=cnot, edge_order=[-1])
carry_r = (
    Qinit(4)
    + Qmotif(E=[(0, 2, 3)], mapping=toffoli)
    + Qmotif(E=[(1, 2)], mapping=cnot)
    + Qmotif(E=[(1, 2, 3)], mapping=toffoli)
)

# ====== Motifs level 2
carry_layer = Qcycle(1, 3, 0, mapping=carry, boundary="open")
carry_r_sum = (
    Qinit(4)
    + Qpivot("1*", merge_within="1111", mapping=carry_r)
    + Qpivot("1*", merge_within="111", mapping=sum)
)
cnot_sum = (
    Qinit(3)
    + Qpivot("*1", merge_within="11", mapping=cnot)
    + Qpivot("*1", merge_within="111", mapping=sum)
)

# ====== Motifs level 3
cnot_sum_pivot = Qpivot("*10", merge_within="111", mapping=cnot_sum)
carry_r_sum_layer = (
    Qmask("*111")
    + Qcycle(1, 3, 0, boundary="open", mapping=carry_r_sum, edge_order=[-1])
    + Qunmask("previous")
)

# ====== Motifs level 4
addition = carry_layer + cnot_sum_pivot + carry_r_sum_layer
n = 4
# plot_circuit(Qinit(2*(n+1))+addition)


# ====== Create problem
# Make bitstrings randomly
d1 = np.random.randint(0, 512)
d2 = np.random.randint(0, 512)
print(d1, d2)
x = [int(b) for b in bin(d1)[2:]]
y = [int(b) for b in bin(d2)[2:]]
max_bits = max(len(x), len(y))
nq = 2 * max_bits + 1
n_anc = max_bits
x = [0] * (max_bits - len(x)) + x
y = [0] * (max_bits - len(y)) + y

# Make names (optional)
ket0 = np.array([1, 0])
ket1 = np.array([0, 1])
tensors = [
    ket0 if elem == 0 else ket1
    for triplet in zip([0] * max_bits, x[::-1], y[::-1])
    for elem in triplet
]
tensors.append(ket0)

circuit = Qinit(len(tensors), tensors=tensors) + addition
result_tensor = circuit()
result = "".join([str(bit[0]) for bit in np.where(result_tensor == 1)])
answer = result[-1] + result[len(result) - 2 : 0 : -3]
print(f"{answer}\n")
print(f"{d1} + {d2} = {int(answer,2)}")
print(result)
