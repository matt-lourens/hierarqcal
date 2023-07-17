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

# n = 10
# N = 2**(2*(n)-3)
# random_int = 1#np.random.randint(0, 2**n)
# Target_string = bin(random_int)[2:].zfill(n)
# N_iterations = 20
# print('Number of qubits', n)
# print('Number of ancillas', 2*(n)-3-n)
# print('Total number of qubits', 2*(n)-3)
# print('Search space size', 2**n)
# print('With target',Target_string)

# print('\nInteractions of Grover to perform', N_iterations)
# print('Optimal number of iterations', int((np.pi/2/np.arctan(1/np.sqrt(2**n))-1)/2))
# # Ration around the zero state but an angle of pi
# H = Qunitary("H()^0")
# X = Qunitary("X()^0")
# H_bottom = Qpivot(mapping=H, global_pattern="*1")

# U_psi = Qcycle(mapping=H)
# U_T = Qpivot(mapping=X, global_pattern=Target_string)

# U_t = Qpivot(mapping=H, global_pattern="*1")
# U_t += Qpivot(
#     mapping=Qunitary("cp(x)^01", symbols=[np.pi / 2]),
#     global_pattern="*1",
#     merge_within="1*",
# )
# U_t += Qpivot(
#     mapping=Qunitary("cnot()^01;cp(x)^12;cnot()^01", symbols=[-np.pi / 2]),
#     global_pattern="*1",
#     merge_within="*1",
# )
# U_t += Qpivot(mapping=H, global_pattern="*1")

# U_toffoli = Qinit(3) + U_t

# maskAncillas = Qmask('0'+'01'*(n-3)+'00')
# multiCZ =  Qcycle(step=2, mapping=U_toffoli, boundary='open') + Qmask('*1') + Qcycle(step = 2, mapping=U_toffoli, edge_order = [-1], boundary='open') + Qunmask('previous') 

# U_rotate = Qcycle(mapping=X)  + H_bottom +  Qunmask('previous') + multiCZ + maskAncillas  + H_bottom + Qcycle(mapping=X)

# U_oracle = U_T + U_rotate + U_T
# U_defuse = U_psi + U_rotate + U_psi

# ancilla_str = '0'+'01'*(n-3)+'00'
# q_names = [f'q_{i}' if ancilla_str[i]== '0' else f'a_{i}' for i in range(2*n-3)]
# if N_iterations>0:
#     U = Qinit(q_names) + maskAncillas + U_psi +  ( U_oracle + U_defuse)*N_iterations 
# else:
#     U = Qinit(q_names) + maskAncillas + U_psi
# # create the circuit using the chose backend
# circuit = U(backend="qiskit", barriers=True)#circuit.copy()
# circuit.measure_all()
# circuit.draw("mpl", fold=50)

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
