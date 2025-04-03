import os
import numpy as np
from numpy import exp as e, cos as c, sin as s
from functools import reduce, partial
import sympy as sp
from hierarqcal import Qunitary, get_tensor_as_f
import scipy.linalg as la

# Helpers for constructing tensor products
def Opi(op, L, i):
    return reduce(np.kron, [np.eye(2) if i != k else op for k in range(L)])

def Opall(op, L):
    return reduce(np.kron, [op for _ in range(L)])
PATH = os.path.dirname(__file__)
i = 1j
# Define Pauli matrices and Identity
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

# Define all two-qubit Pauli products
XX = np.kron(X, X)
YY = np.kron(Y, Y)
ZZ = np.kron(Z, Z)
XY = np.kron(X, Y)
XZ = np.kron(X, Z)
YX = np.kron(Y, X)
YZ = np.kron(Y, Z)
ZX = np.kron(Z, X)
ZY = np.kron(Z, Y)

# Parameterized two-qubit entanglers for each combination
XXe = lambda θ: la.expm(1j * θ * XX)
YYe = lambda θ: la.expm(1j * θ * YY)
ZZe = lambda θ: la.expm(1j * θ * ZZ)
XYe = lambda θ: la.expm(1j * θ * XY)
XZe = lambda θ: la.expm(1j * θ * XZ)
YXe = lambda θ: la.expm(1j * θ * YX)
YZe = lambda θ: la.expm(1j * θ * YZ)
ZXe = lambda θ: la.expm(1j * θ * ZX)
ZYe = lambda θ: la.expm(1j * θ * ZY)

# Parameterized partial-SWAP gate
def p_swap(theta):
    swap = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    return la.expm(1j * theta * swap)

# Define fixed two-qubit gates
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
])
sqrt_SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0.5*(1+1j), 0.5*(1-1j), 0],
    [0, 0.5*(1-1j), 0.5*(1+1j), 0],
    [0, 0, 0, 1]
])
iSWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1j, 0],
    [0, 1j, 0, 0],
    [0, 0, 0, 1]
])

# Other single-qubit gates and helpers
H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
Ii, Xi, Yi, Zi = partial(Opi, I), partial(Opi, X), partial(Opi, Y), partial(Opi, Z)
R = lambda O, θ: c(θ/2)*I - 1j * s(θ/2)*O
ket0 = np.array([1, 0])
ket1 = np.array([0, 1])
Uz2x = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
Uz2y = 1 / np.sqrt(2) * np.array([[1, -1j], [1, 1j]])
H_annni = lambda N, k, h: (
    -1/4 * reduce(np.add, [Zi(N, i) @ Zi(N, (i + 1) % N) for i in range(N)])
    + k/4 * reduce(np.add, [Zi(N, i) @ Zi(N, (i + 2) % N) for i in range(N)])
    - h/2 * reduce(np.add, [Xi(N, i) for i in range(N)])
)
H_annni_norm = lambda N, k, h: (
    -reduce(np.add, [Zi(N, i) @ Zi(N, (i + 1) % N) for i in range(N)])
    + k*reduce(np.add, [Zi(N, i) @ Zi(N, (i + 2) % N) for i in range(N)])
    - h*reduce(np.add, [Xi(N, i) for i in range(N)])
)
H_ising= lambda N=3, h=.5: (
    -1/4 * reduce(np.add, [Zi(N, i) @ Zi(N, (i + 1) % N) for i in range(N)])
    - h/2 * reduce(np.add, [Xi(N, i) for i in range(N)])
)
rx = partial(R, X)
ry = partial(R, Y)
rz = partial(R, Z)
crx = lambda θ: np.kron(np.outer(ket0, ket0), I) + np.kron(np.outer(ket1, ket1), rx(θ))
cry = lambda θ: np.kron(np.outer(ket0, ket0), I) + np.kron(np.outer(ket1, ket1), ry(θ))
crz = lambda θ: np.kron(np.outer(ket0, ket0), I) + np.kron(np.outer(ket1, ket1), rz(θ))

# Three-qubit entanglers
# (a) A symmetric three-body entangler: exp[iθ(X⊗X⊗X + Y⊗Y⊗Y + Z⊗Z⊗Z)]
XXX = np.kron(np.kron(X, X), X)
YYY = np.kron(np.kron(Y, Y), Y)
ZZZ = np.kron(np.kron(Z, Z), Z)
three_body_op = XXX + YYY + ZZZ
three_ent = lambda θ: la.expm(1j * θ * three_body_op)

# Wrap gates into Qunitary objects
qh      = Qunitary(get_tensor_as_f(H), arity=1, n_symbols=0, name="h")
qx      = Qunitary(get_tensor_as_f(X), arity=1, n_symbols=0, name="x")
qy      = Qunitary(get_tensor_as_f(Y), arity=1, n_symbols=0, name="y")
qz      = Qunitary(get_tensor_as_f(Z), arity=1, n_symbols=0, name="z")
qrx     = Qunitary(get_tensor_as_f(rx), n_symbols=1, arity=1, name="rx")
qry     = Qunitary(get_tensor_as_f(ry), n_symbols=1, arity=1, name="ry")
qrz     = Qunitary(get_tensor_as_f(rz), n_symbols=1, arity=1, name="rz")
qcrx    = Qunitary(get_tensor_as_f(crx), n_symbols=1, arity=2, name="crx")
qcry    = Qunitary(get_tensor_as_f(cry), n_symbols=1, arity=2, name="cry")
qcrz    = Qunitary(get_tensor_as_f(crz), n_symbols=1, arity=2, name="crz")
qXXe    = Qunitary(get_tensor_as_f(XXe), n_symbols=1, arity=2, name="XXe")
qYYe    = Qunitary(get_tensor_as_f(YYe), n_symbols=1, arity=2, name="YYe")
qZZe    = Qunitary(get_tensor_as_f(ZZe), n_symbols=1, arity=2, name="ZZe")
qXYe    = Qunitary(get_tensor_as_f(XYe), n_symbols=1, arity=2, name="XYe")
qXZe    = Qunitary(get_tensor_as_f(XZe), n_symbols=1, arity=2, name="XZe")
qYXe    = Qunitary(get_tensor_as_f(YXe), n_symbols=1, arity=2, name="YXe")
qYZe    = Qunitary(get_tensor_as_f(YZe), n_symbols=1, arity=2, name="YZe")
qZXe    = Qunitary(get_tensor_as_f(ZXe), n_symbols=1, arity=2, name="ZXe")
qZYe    = Qunitary(get_tensor_as_f(ZYe), n_symbols=1, arity=2, name="ZYe")
qCNOT   = Qunitary(get_tensor_as_f(CNOT), arity=2, n_symbols=0, name="cnot")
qpswap  = Qunitary(get_tensor_as_f(p_swap), n_symbols=1, arity=2, name="pswap")
qiSWAP  = Qunitary(get_tensor_as_f(iSWAP), arity=2, n_symbols=0, name="iswap")
qSqrtSwap = Qunitary(get_tensor_as_f(sqrt_SWAP), arity=2, n_symbols=0, name="sqrt_swap")
q3ent   = Qunitary(get_tensor_as_f(three_ent), n_symbols=1, arity=3, name="3ent")

# Final hierarchical gate list (all in one line)

