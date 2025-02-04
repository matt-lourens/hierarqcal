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
    Qhierarchy,
)
import numpy as np
from numpy import exp as e, cos as c, sin as s, pi as π

i = 1j
initial_tensors = [
    np.array([0, 1]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0]),
]
cry = lambda θ: np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, c(θ / 2), -s(θ / 2)],
        [0, 0, s(θ / 2), c(θ / 2)],
    ]
).reshape(2, 2, 2, 2)
ry = lambda θ: np.array([[c(θ / 2), -s(θ / 2)], [s(θ / 2), c(θ / 2)]]).reshape(2, 2)


def get_tensor_as_f1(u):
    def generic_f(bits, symbols=None, state=None, u=u):
        # bits not acted upon
        orig_shape = state.shape
        nbits = tuple([k for k in range(len(orig_shape)) if k not in bits])
        nbits_size = np.product([orig_shape[k] for k in nbits])
        bits_size = np.product([orig_shape[k] for k in bits])
        # put bits not acted on last
        perm = bits + nbits
        perminv = [perm.index(k) for k in range(len(perm))]
        state = state.transpose(perm)
        # turn into matrix
        state = state.reshape(bits_size, nbits_size)
        um = u(*symbols).reshape(bits_size, bits_size)
        state = um @ state
        state = state.reshape(orig_shape)
        state = state.transpose(perminv)
        return state

    return generic_f



qcry = Qunitary(get_tensor_as_f1(cry), n_symbols=1, arity=2)
qry = Qunitary(get_tensor_as_f1(ry), n_symbols=1, arity=1)




"""
Testing
"""
from functools import reduce, partial


def Opi(op, L, i):
    return reduce(np.kron, [np.eye(2) if not (i == k) else op for k in range(L)])


def Opall(op, L):
    return reduce(np.kron, [op for k in range(1, L + 1)])


I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
Ii, Xi, Yi, Zi = partial(Opi, I), partial(Opi, X), partial(Opi, Y), partial(Opi, Z)
Ry = lambda θ: np.cos(θ / 2) * I - 1j * np.sin(θ / 2) * Y
ket0 = np.array([1, 0])
ket1 = np.array([0, 1])
Uz2y = 1 / np.sqrt(2) * np.array([[1, -1j], [1, 1j]])

def Cry(L, i, j, θ):
    A = reduce(
        np.kron,
        [np.outer(ket0, ket0) if i == k else np.eye(2) for k in range(L)],
    )
    B = reduce(
        np.kron,
        [
            np.outer(ket1, ket1) if i == k else (Ry(θ) if j == k else np.eye(2))
            for k in range(L)
        ],
    )
    return A + B


# state = reduce(np.kron, initial_tensors)

# ψt = Cry(4, 1, 3, θ) @ ψ0.reshape(2**4)
θ, φ = 0.5, 0.3
initial_tensors = [
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0]),
    np.array([1, 0]),
]
N = len(initial_tensors)
motif = Qinit(tensors=initial_tensors) + Qmotif(E=[(0,1)], mapping=qcry) #+ Qcycle(mapping=qcry)
motif.set_symbols([θ])
ψ1 = motif().reshape(2**N)
Opall( Uz2y,N)@ψ1
print("oi")
