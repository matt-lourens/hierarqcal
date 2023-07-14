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


n = 5
N = 2 ** (2 * (n) - 3)
random_int = np.random.randint(0, N)
target_string = bin(random_int)[2:].zfill(n-3)
print(int((np.pi / 2 * np.sqrt(N) - 1) / 2), "interactions of Grover")
print("With target", target_string)
print("Search space size", N, "Qubit", 2 * (n) - 3)
# Ration around the zero state but an angle of pi
H = Qunitary("H()^0")
X = Qunitary("X()^0")
H_bottom = Qpivot(mapping=H, global_pattern="*1")

U_psi = Qcycle(mapping=H)
U_T = Qpivot(mapping=X, global_pattern=target_string)
U_t = Qpivot(mapping=H, global_pattern="*1")
U_t += Qpivot(
    mapping=Qunitary("cp(x)^01", symbols=[np.pi / 2]),
    global_pattern="*1",
    merge_within="1*",
)
U_t += Qpivot(
    mapping=Qunitary("cnot()^01;cp(x)^12;cnot()^01", symbols=[np.pi / 2]),
    global_pattern="*1",
    merge_within="*1",
)
U_t += Qpivot(mapping=H, global_pattern="*1")

U_toffoli = Qinit(3) + U_t


maskAncillas = Qmask("0" + "01" * (n - 3) + "00")
multiCZ = (
    Qcycle(step=2, mapping=U_toffoli, boundary="open")
    + Qmask("*1")
    + Qcycle(
        step=2, mapping=U_toffoli, share_weights=True, boundary="open", edge_order=[-1]
    )
    + Qunmask("previous")
)

U_rotate = H_bottom + Qunmask("previous") + multiCZ + maskAncillas + H_bottom

U_oracle = U_T + U_rotate + U_T
U_defuse = U_psi + U_rotate + U_psi

ancilla_str = "0" + "01" * (n - 3) + "00"
q_names = [f"q_{i}" if ancilla_str[i] == "0" else f"a_{i}" for i in range(2 * n - 3)]
U = (
    Qinit(q_names)
    + U_psi
    + (maskAncillas + U_oracle + U_defuse) * int((np.pi / 2 * np.sqrt(N) - 1) / 2)
)  # int((np.pi/2*np.sqrt(N)-1)/2)

circuit = U(backend="qiskit", barriers=False)
circuit.draw("mpl")
# Add measurements
circuit.measure_all()

# execute circuit
from qiskit import Aer, execute

shots=1000
aer_backend = Aer.get_backend("qasm_simulator")
job = execute(circuit, aer_backend, shots=shots)
result = job.result()
counts = result.get_counts(circuit)

print(f"Counts: {counts}")
# get most likely result
most_likely = result.get_counts(circuit).most_frequent()
output_wo_ancilla = ''.join([most_likely[i]  for i in range(2 * n - 3) if ancilla_str[i] == "1" ] )
# Get count of most likely
most_likely_count = result.get_counts(circuit).get(most_likely)
print(
    f"target string: {target_string}\n returned string: {output_wo_ancilla}\n counts: {most_likely_count}\n shots {shots}"
)

circuit.draw("mpl")


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
#             merge_within="01",
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
#             merge_within="01",
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
