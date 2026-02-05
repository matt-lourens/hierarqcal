# %%
import os
import re
from collections import namedtuple
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from hierarqcal import (
    Qhierarchy,
    Qcycle,
    Qpermute,
    Qmask,
    Qunmask,
    Qpivot,
    Qinit,
    Qmotif,
    Qmotifs,
    plot_motif,
    plot_circuit,
    Qunitary,
    get_quimb_as_f,
)
from autoray import do, backend_like
import quimb as qu
import quimb.tensor as qtn
import dill

# import pennylane as qml
import re
from itertools import product
import json
import shelve
import jax.numpy as jnp
import autoray


def COe(O0):
    # static (numpy/quimb) objects – independent of backend
    I2 = qu.identity(2)
    Z = qu.pauli("Z")
    P0 = (I2 + Z) / 2  # |0><0|
    P1 = (I2 - Z) / 2  # |1><1|
    σO = qu.pauli(O0)  # target Pauli

    def generic_unitary(θs):
        θ = θs[0]  # scalar symbol / tensor
        with backend_like(θ):  # makes everything dtype- & device-aware
            zero = θ * 0.0
            c = do("complex", do("cos", θ / 2), zero)  #  cos θ
            s = do("complex", zero, do("sin", θ / 2))  # i·sin θ

            I2_b = do("array", I2, like=θ, dtype=c.dtype)
            σO_b = do("array", σO, like=θ, dtype=c.dtype)
            P0_b = do("array", P0, like=θ, dtype=c.dtype)
            P1_b = do("array", P1, like=θ, dtype=c.dtype)

            U_t = c * I2_b - s * σO_b  # e^{iθσ_O}
            U = do("kron", P0_b, I2_b, like=θ) + do("kron", P1_b, U_t, like=θ)
            # U = qu.kron(P0_b, I2_b) + qu.kron(P1_b, U_t)

            return do("reshape", U, (2, 2, 2, 2))  # (ctl_in, tgt_in, ctl_out, tgt_out)

    return generic_unitary


def prodOs(Os):
    def generic_unitary(θs):
        θ = θs[0]
        with backend_like(θ):
            zero = θ * 0.0
            one = zero + 1.0  # Ensure it's a tensor type
            img = do("complex", zero, one)
            I = do("array", qu.identity(2), like=θ, dtype=img.dtype)
            terms = []
            if len(Os) == 1:
                G = do(
                    "array", 1 / 2 * θs[0] * qu.pauli(Os[0]), like=θ, dtype=img.dtype
                )
            else:
                for ind, O in enumerate(Os):
                    oplist = [I] * len(Os)
                    oplist[ind] = do(
                        "array", 1 / 2 * qu.pauli(O), like=θ, dtype=img.dtype
                    )
                    term = reduce(lambda A, B: do("kron", A, B, like=θ), oplist)
                    terms.append(term)
                G = θs[0] * reduce(lambda A, B: do("matmul", A, B, like=θ), terms)
            U = do("linalg.expm", -img * G)
            return do("reshape", U, (2,) * (2 * len(Os)))

    return generic_unitary


def sumOs(Os, share_symbols=False):
    def generic_unitary(θs):
        θ = θs[0]
        if share_symbols is True:
            θs = [θ] * len(Os)
        with backend_like(θ):
            zero = θ * 0.0
            one = zero + 1.0  # Ensure it's a tensor type
            img = do("complex", zero, one)
            I = do("array", qu.identity(2), like=θ, dtype=img.dtype)
            terms = []
            for ind, O in enumerate(Os):
                oplist = [I] * len(Os)
                oplist[ind] = do("array", 1 / 2 * qu.pauli(O), like=θ, dtype=img.dtype)
                term = reduce(lambda A, B: do("kron", A, B, like=θ), oplist)
                terms.append(θs[ind] * term)
            U = do("linalg.expm", -img * sum(terms))  # sum of terms
            return do("reshape", U, (2,) * (2 * len(Os)))

    return generic_unitary


MAPPING_DICT = {}
MAX_PAULISTRING_LEN = 3
pauli_strings = [
    "".join(pauli_string)
    for reps in range(1, MAX_PAULISTRING_LEN + 1)
    for pauli_string in product(["I", "X", "Y", "Z"], repeat=reps)
]
hierq_gates = []
for lbl in pauli_strings:
    varname = "".join(["e"] + [f"{char}" for char in lbl])
    globals()[varname] = Qunitary(
        get_quimb_as_f(prodOs(lbl)), n_symbols=1, arity=len(lbl), name=varname
    )
    MAPPING_DICT[varname] = globals()[varname]
    hierq_gates.append(globals()[varname])
    if len(lbl) > 1:
        # sums that don't share symbol
        varname = "".join(["e"] + [f"{char}p" for char in lbl])[:-1:]
        globals()[varname] = Qunitary(
            get_quimb_as_f(sumOs(lbl, share_symbols=False)),
            n_symbols=len(lbl),
            arity=len(lbl),
            name=varname,
        )
        MAPPING_DICT[varname] = globals()[varname]
        hierq_gates.append(globals()[varname])
        # sums that share symbol
        varname = "".join(["se"] + [f"{char}p" for char in lbl])[:-1:]
        globals()[varname] = Qunitary(
            get_quimb_as_f(sumOs(lbl, share_symbols=True)),
            n_symbols=1,
            arity=len(lbl),
            name=varname,
        )
        MAPPING_DICT[varname] = globals()[varname]
        hierq_gates.append(globals()[varname])

# Controlled rotations
for lbl in ["X", "Y", "Z"]:
    varname = f"cr{lbl}"
    globals()[varname] = Qunitary(
        get_quimb_as_f(COe(lbl)), n_symbols=1, arity=2, name=varname
    )
    MAPPING_DICT[varname] = globals()[varname]
    hierq_gates.append(globals()[varname])


def load_motif_cache(DISK_CACHE_PATH):
    """
    Loads the entire on-disk cache into a Python dictionary.
    Call this once when your program starts.

    Returns:
        dict: A dictionary containing the entire cache.
    """
    # The 'c' flag will create the file if it doesn't exist.
    with shelve.open(DISK_CACHE_PATH, flag="c") as disk_cache:
        # dict() efficiently converts the entire shelve object to a standard dictionary
        return dict(disk_cache)


# %%
def get_LMG_MPO(J: float, h: float, N: int):
    """
    Open-boundary MPO for the Lipkin-Meshkov-Glick model

        H = -J/(4N) * sum_{i<j} Z_i Z_j  - (h/2) * sum_i X_i

    Returns
    -------
    H0_local : (D, 2, 2)      left-boundary tensor
    Hb_local : (D, D, 2, 2)   bulk tensor
    HN_local : (D, 2, 2)      right-boundary tensor
    with bond dimension D = 3.
    """
    # Pauli matrices
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    ZERO = np.zeros_like(I)

    # Helpful constants
    c_const = 0#J / (8 * N)  # gives the +J/8 overall shift
    z_pair = -J / (4 * N)  # prefactor for each Z_i Z_j
    x_field = -h / 2  # prefactor for each X_i

    # ---- bulk MPO tensor  W  (shape 3×3, each entry is a 2×2 matrix) ----
    Hb = np.array(
        [
            #  col:    0           1               2
            [
                [I, Z, x_field * X + c_const * I],  # row 0
                [ZERO, I, z_pair * Z],  # row 1
                [ZERO, ZERO, I],
            ]  # row 2
        ],
    )[
        0
    ]  # strip the redundant first axis added by np.array

    H0 = np.array([Hb[0, 0], Hb[0, 1], Hb[0, 2]])   # start in state 0
    HN = np.array([Hb[0, 2], Hb[1, 2], Hb[2, 2]]) 

    return H0, Hb, HN


# %%
SIZES = [6]
N = SIZES[0]
H_RANGE = np.linspace(0, 1, 20)  # range of h values
ALL_MPOs = {}
inds = []
for size in SIZES:
    for hind, hv in enumerate(H_RANGE):
        H0, Hb, HN = get_LMG_MPO(1, hv, size)
        mpo = qtn.MatrixProductOperator([H0] + [Hb] * (size - 2) + [HN])
        mpo = mpo.astype("complex128")
        ind = hind
        inds.append(ind)
        ALL_MPOs[(size, ind)] = mpo
raw_bank = []
for ind in inds:
    raw, _ = qtn.pack(ALL_MPOs[(N, ind)])
    raw_bank.append({k: jnp.asarray(v) for k, v in raw.items()})
keys = sorted(raw_bank[0].keys())
stacked = {k: jnp.stack([rb[k] for rb in raw_bank]) for k in keys}
_, mpo_skeleton = qtn.pack(ALL_MPOs[(N, 0)])


def make_mpo(ind):  # ind is 0-D int32 JAX array
    raw = {k: stacked[k][ind] for k in keys}  # numeric pytree
    return qtn.unpack(raw, mpo_skeleton)

def norm_fn(psi):
    n = (psi.H @ psi) ** 0.5      # ‖ψ‖
    return psi.multiply(1/n, spread_over='all')

def energy(psi, ind):
    mpo = make_mpo(ind)  # ← ordinary MPO object
    psi_h = psi.H
    psi.align_(mpo, psi_h)
    E_cplx = (psi_h & mpo & psi).contract(all, optimize="auto-hq")
    return autoray.do("real", E_cplx)


def get_contraction_width(psi, ind):
    mpo = make_mpo(ind)  # ← ordinary MPO object
    psi_h = psi.H
    psi.align_(mpo, psi_h)
    E_cplx = psi_h & mpo & psi
    con_tree = E_cplx.contraction_tree(optimize="auto-hq")
    return con_tree.contraction_width()


# %%
hierq = (
    Qinit(N, state=qtn.Circuit(N))
    + Qcycle(mapping=eY, boundary="periodic")
    # + Qcycle(mapping=crY, boundary="periodic")
)
param_vals = [0.3, 0.5]
params = [
    {"name": f"x{i}", "val": jnp.array(val, dtype=jnp.float64)}
    for i, val in enumerate(param_vals)
]
hierq.set_symbols(params)
circ = hierq(backend="quimb")
psi = circ.psi
# %%
for hind, hv in enumerate(H_RANGE):
    print(energy(psi, hind))
# %%
ψ0 = np.zeros(2**N, dtype=np.complex128)
for ind, n in enumerate(range(2**6)):
    ψ0[ind] = circ.amplitude(f"{n:06b}")
# %%
from utils import *

H_LMG = lambda N=3, h=0.5: (
    -1
    / (4 * N)
    * reduce(np.add, [Zi(N, i) @ Zi(N, j) for i in range(N) for j in range(N) if i < j])
    - h / 2 * reduce(np.add, [Xi(N, i) for i in range(N)])
)
motif = Qcycle(mapping=qYe, boundary="periodic")
# + Qcycle(mapping=qcry, boundary="periodic"))
hierq = Qinit(tensors=[ket0] * N) + motif
hierq.set_symbols(param_vals)
ψ = hierq().reshape(-1)
for hv in H_RANGE:
    en = np.conj(ψ) @ (H_LMG(N, hv) @ ψ)
    print(en)

# %%
