# %%
import os
os.environ["JAX_ENABLE_X64"] = "True"
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
import jax
import jax.numpy as jnp
import autoray
from itertools import combinations, product
from functools import reduce

jax.config.update("jax_enable_x64", True)
JAX_DTYPE = jnp.float64
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
            X = qu.pauli("X")
            Y = qu.pauli("Y")
            Z = qu.pauli("Z")
            P = (X + img*Y) / 2 
            M = (X - img*Y) / 2 
            terms = []
            if len(Os) == 1:
                if Os[0] == "P":
                    Op = P
                elif Os[0] == "M":
                    Op = M
                else:
                    Op = qu.pauli(Os[0])
                G = do(
                    "array", 1 / 2 * θs[0] * Op, like=θ, dtype=img.dtype
                )
            else:
                for ind, O in enumerate(Os):
                    oplist = [I] * len(Os)
                    if O == "P":
                        Op = P
                    elif O == "M":
                        Op = M
                    else:
                        Op = qu.pauli(O)
                    oplist[ind] = do(
                        "array", 1 / 2 * Op, like=θ, dtype=img.dtype
                    )
                    term = reduce(lambda A, B: do("kron", A, B, like=θ), oplist)
                    terms.append(term)
                G = θs[0] * reduce(lambda A, B: do("matmul", A, B, like=θ), terms)
            U = do("linalg.expm", -img * G)
            return do("reshape", U, (2,) * (2 * len(Os)))

    return generic_unitary

# ------------------------------------------------------------------
# Raising/lowering version of `prodOs`
# ------------------------------------------------------------------
def prodPMs(Os):
    def generic_unitary(θs):
        θ = θs[0]
        with backend_like(θ):
            zero = θ * 0.0
            one = zero + 1.0  # Ensure it's a tensor type
            img = do("complex", zero, one)
            I = do("array", qu.identity(2), like=θ, dtype=img.dtype)
            X = qu.pauli("X")
            Y = qu.pauli("Y")
            Z = qu.pauli("Z")
            P = (X + img*Y) / 2 
            M = (X - img*Y) / 2 
            terms = []
            if len(Os) == 1:
                if Os[0] == "P":
                    Op = P
                elif Os[0] == "M":
                    Op = M
                else:
                    Op = qu.pauli(Os[0])
                G = do(
                    "array", θs[0] * Op, like=θ, dtype=img.dtype
                )
            else:
                for ind, O in enumerate(Os):
                    oplist = [I] * len(Os)
                    if O == "P":
                        Op = P
                    elif O == "M":
                        Op = M
                    else:
                        Op = qu.pauli(O)
                    oplist[ind] = do(
                        "array",  Op, like=θ, dtype=img.dtype
                    )
                    term = reduce(lambda A, B: do("kron", A, B, like=θ), oplist)
                    terms.append(term)
                G = θs[0] * reduce(lambda A, B: do("matmul", A, B, like=θ), terms)
            U = do("linalg.expm", G)
            return do("reshape", U, (2,) * (2 * len(Os)))

    return generic_unitary


def sumOs(Os, share_symbols=False):
    """
    Os = ["X","Y","X"] -> e^(X+Y+Z)
    Os = ["XY","ZY"] -> e^(XY+ZY)
    all strings in list must be same length
    """

    def generic_unitary(θs):
        θ = θs[0]
        if share_symbols is True:
            θs = [θ] * len(Os)
        with backend_like(θ):
            zero = θ * 0.0
            one = zero + 1.0
            img = do("complex", zero, one)
            I = do("array", qu.identity(2), like=θ, dtype=img.dtype)
            terms = []
            nq = len(Os[0])  # assuming all are same length
            for ind1, Ostring in enumerate(Os):
                oplist = [I] * len(Ostring)
                for ind2, O in enumerate(Ostring):
                    oplist[ind2] = do(
                        "array", 1 / 2 * qu.pauli(O), like=θ, dtype=img.dtype
                    )
                term = reduce(lambda A, B: do("kron", A, B, like=θ), oplist)
                terms.append(θs[ind1] * term)
            U = do("linalg.expm", -img * sum(terms))
            return do("reshape", U, (2,) * (2 * nq))

    return generic_unitary
# %%
ALL_GATES = True
MAX_PAULISTRING_LEN = 2
GATES = []
MAPPING_DICT = {}
pauli_strings = [
    "".join(pauli_string)
    for reps in range(1, MAX_PAULISTRING_LEN + 1)
    for pauli_string in product(["I", "X", "Y", "Z"], repeat=reps)
]
pm_strings = [
    "".join(pauli_string)
    for reps in range(1, MAX_PAULISTRING_LEN + 1)
    for pauli_string in product(["I", "Z", "P","M"], repeat=reps)
]
one_length = [ps for ps in pauli_strings if len(ps) == 1]
two_length = [ps for ps in pauli_strings if len(ps) == 2]
three_length = [ps for ps in pauli_strings if len(ps) == 3]

one_length_pm = [ps for ps in pm_strings if len(ps) == 1]
two_length_pm = [ps for ps in pm_strings if len(ps) == 2]
three_length_pm = [ps for ps in pm_strings if len(ps) == 3]

one_combos = [
    combo for reps in range(2, 3) for combo in combinations(one_length, r=reps)
]

two_combos = [
    combo for reps in range(2, 3) for combo in combinations(two_length, r=reps)
]

one_combos_pm = [
    combo for reps in range(2, 3) for combo in combinations(one_length_pm, r=reps)
]

two_combos_pm = [
    combo for reps in range(2, 3) for combo in combinations(two_length_pm, r=reps)
]
# one_combos = product(one_length, repeat=2)
hierq_gates = []
for lbl in pauli_strings:
    varname = "".join(["e"] + [f"{char}" for char in lbl])
    globals()[varname] = Qunitary(
        get_quimb_as_f(prodOs(lbl)), n_symbols=1, arity=len(lbl), name=varname
    )
    MAPPING_DICT[varname] = globals()[varname]
    if varname in GATES or ALL_GATES:
        hierq_gates.append(globals()[varname])

for Os in one_combos:
    varname = "".join(["e"] + ["p".join([lbl for lbl in Os])])
    globals()[varname] = Qunitary(
        get_quimb_as_f(sumOs(Os, share_symbols=False)),
        n_symbols=len(Os),
        arity=len(Os[0]),
        name=varname,
    )
    MAPPING_DICT[varname] = globals()[varname]
    if varname in GATES or ALL_GATES:
        hierq_gates.append(globals()[varname])

for Os in two_combos:
    varname = "".join(["e"] + ["p".join([lbl for lbl in Os])])
    globals()[varname] = Qunitary(
        get_quimb_as_f(sumOs(Os, share_symbols=False)),
        n_symbols=len(Os),
        arity=len(Os[0]),
        name=varname,
    )
    MAPPING_DICT[varname] = globals()[varname]
    if varname in GATES or ALL_GATES:
        hierq_gates.append(globals()[varname])


hierq_gates = []
for lbl in pm_strings:
    varname = "".join(["e"] + [f"{char}" for char in lbl])
    globals()[varname] = Qunitary(
        get_quimb_as_f(prodPMs(lbl)), n_symbols=1, arity=len(lbl), name=varname
    )
    MAPPING_DICT[varname] = globals()[varname]
    if varname in GATES or ALL_GATES:
        hierq_gates.append(globals()[varname])

# Controlled rotations
for lbl in ["X", "Y", "Z"]:
    varname = f"cr{lbl}"
    globals()[varname] = Qunitary(
        get_quimb_as_f(COe(lbl)), n_symbols=1, arity=2, name=varname
    )
    MAPPING_DICT[varname] = globals()[varname]
    if varname in GATES or ALL_GATES:
        hierq_gates.append(globals()[varname])

# %%
N=5
hierq = (
    Qinit(N, state=qtn.Circuit(N))
    + Qcycle(mapping=eP, boundary="open")
    + Qcycle(mapping=eZ, boundary="open")
    + Qcycle(mapping=eM, boundary="open")
)
    # + Qcycle(mapping=crY, boundary="periodic")
θ = 0.3
param_vals = [-np.tan(θ/2), np.log(np.cos(θ/2)), np.tan(θ/2)]
params = [
    {"name": f"x{i}", "val": jnp.array(val, dtype=jnp.float64)}
    for i, val in enumerate(param_vals)
]
plot_circuit(hierq)
hierq.set_symbols(params)
circ = hierq(backend="quimb")
psi = circ.psi
ψ1 = np.zeros(2**N, dtype=np.complex128)
for ind, n in enumerate(range(2**5)):
    ψ1[ind] = circ.amplitude(f"{n:05b}")
ψ1
# %%
N=5
hierq = (
    Qinit(N, state=qtn.Circuit(N))
    + Qcycle(mapping=eY, boundary="periodic")
    # + Qcycle(mapping=crY, boundary="periodic")
)
param_vals = [0.3]
params = [
    {"name": f"x{i}", "val": jnp.array(val, dtype=jnp.float64)}
    for i, val in enumerate(param_vals)
]
hierq.set_symbols(params)
circ = hierq(backend="quimb")
psi = circ.psi
ψ0 = np.zeros(2**N, dtype=np.complex128)
for ind, n in enumerate(range(2**5)):
    ψ0[ind] = circ.amplitude(f"{n:05b}")
ψ0
# %%
# %%
N=7
# sub = (Qinit(2)+ Qcycle(mapping=eZP, boundary="open")
#     + Qcycle(mapping=eIZ, boundary="open")
#     + Qcycle(mapping=eZM, boundary="open"))
# hierq = (
#     Qinit(N, state=qtn.Circuit(N))
#     + Qcycle(mapping=sub, boundary="periodic")
# )
# hierq = (
#     Qinit(N, state=qtn.Circuit(N))
#     + Qcycle(mapping=eP, boundary="open")
#     + Qcycle(mapping=eZ, boundary="open")
#     + Qcycle(mapping=eM, boundary="open"))

#     # + Qcycle(mapping=crY, boundary="periodic")
# θ = 0.3
# param_vals = [-np.tan(θ/4), np.log(np.cos(θ/4)), np.tan(θ/4)]
# params = [
#     {"name": f"x{i}", "val": jnp.array(val, dtype=jnp.float64)}
#     for i, val in enumerate(param_vals)
# ]
# plot_circuit(hierq)
# hierq.set_symbols(params)
# circ = hierq(backend="quimb")
# psi = circ.psi
# ψ1 = np.zeros(2**N, dtype=np.complex128)
# for ind, n in enumerate(range(2**N)):
#     ψ1[ind] = circ.amplitude(f"{n:05b}")
# ψ1
# %%
# N=5
# hierq = (
#     Qinit(N, state=qtn.Circuit(N))
#     + Qcycle(mapping=eZY, boundary="periodic")
#     # + Qcycle(mapping=crY, boundary="periodic")
# )
# param_vals = [0.3]
# params = [
#     {"name": f"x{i}", "val": jnp.array(val, dtype=jnp.float64)}
#     for i, val in enumerate(param_vals)
# ]
# hierq.set_symbols(params)
# circ = hierq(backend="quimb")
# psi = circ.psi
# ψ0 = np.zeros(2**N, dtype=np.complex128)
# for ind, n in enumerate(range(2**N)):
#     ψ0[ind] = circ.amplitude(f"{n:05b}")
# ψ0

# %%
hierq = (
    Qinit(N, state=qtn.Circuit(N))
    + Qcycle(mapping=eP, boundary="open")
    + Qcycle(mapping=eZ, boundary="open")
    + Qcycle(mapping=eM, boundary="open")
    +Qcycle(mapping=eZP, boundary="periodic")
    + Qcycle(mapping=eZ, boundary="periodic")
    + Qcycle(mapping=eZM, boundary="periodic"))

hierq = (
    Qinit(N, state=qtn.Circuit(N))
    + Qcycle(mapping=eY, boundary="open")
    + Qcycle(mapping=eZY, boundary="periodic"))
plot_circuit(hierq)
symbol_namesvalues = [
    {
        "name": str(sym),
        "val": jnp.array([np.random.rand()], dtype=JAX_DTYPE),
    }
    for sym in hierq.get_symbols()
]
hierq.set_symbols(symbol_namesvalues)
circ = hierq(backend="quimb")
# %%
Nh=10
H_RANGE = np.linspace(0, 1, Nh)
N_ITER = 100
REPS = 3
SIZES = [5,6,7]
def get_ising_mpo_pbc(J, h):
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    ZERO = np.zeros_like(I)
    Hb = np.array(
        [
            [I, ZERO, ZERO],
            [Z, ZERO, ZERO],
            [-h / 2 * X, -J / 4 * Z, I],
        ],
        dtype=float,
    )  # shape (3, 3, 2, 2)
    H0 = np.array(
        [
            [-h / 2 * X, -1 / 4 * J * Z, I],
            [ZERO, ZERO, Z],
            [ZERO, ZERO, ZERO],
        ],
        dtype=float,
    )  # shape (3, 3, 2, 2)
    return H0, Hb

ALL_MPOs = {}
for size in SIZES:
    inds = []
    for hind, hv in enumerate(H_RANGE):
        H0, Hb = get_ising_mpo_pbc(1, hv)
        mpo = qtn.MatrixProductOperator([H0] + [Hb] * (size - 1))
        mpo = mpo.astype("complex128")
        ind = hind
        inds.append(ind)
        ALL_MPOs[(size, ind)] = mpo
pathinds = [hind for hind in range(Nh)]
psi = circ.psi
raw_bank = []
for ind in inds:
    raw, _ = qtn.pack(ALL_MPOs[(N, ind)])
    raw_bank.append({k: jnp.asarray(v) for k, v in raw.items()})
keys = sorted(raw_bank[0].keys())
stacked = {k: jnp.stack([rb[k] for rb in raw_bank]) for k in keys}
_, mpo_skeleton = qtn.pack(ALL_MPOs[(N, 0)])

def make_mpo(ind):
    raw = {k: stacked[k][ind] for k in keys}
    return qtn.unpack(raw, mpo_skeleton)

def energy(psi, ind):
    mpo = make_mpo(ind)
    psi_h = psi.H
    psi.align_(mpo, psi_h)
    E_cplx = (psi_h & mpo & psi).contract(all, optimize="auto-hq")
    return autoray.do("real", E_cplx)

def norm_fn(psi):
    n = (psi.H @ psi) ** 0.5
    return psi.multiply(1/n, spread_over='all')

energies = []
opt = qtn.TNOptimizer(
    psi,
    loss_fn=energy,
    norm_fn= norm_fn,
    # loss_constants={"MPO": MPO},
    tags=[parm["name"] for parm in symbol_namesvalues],
    shared_tags=[parm["name"] for parm in symbol_namesvalues],
    autodiff_backend="jax",  # use 'autograd' for non-compiled optimization
    optimizer="L-BFGS-B",  # the optimization algorithm
    progbar=True
)

# %%
energies = []
for ind in pathinds:
    opt.set_loss_var(jnp.asarray(ind, dtype=jnp.int64))
    _ = opt.optimize(N_ITER)
    energies.append(opt.loss)
# %%
energies = []
for ind in pathinds:
    tmp_energies = []
    opt.set_loss_var(jnp.asarray(ind, dtype=jnp.int64))
    _ = opt.optimize(N_ITER)
    tmp_energies.append(opt.loss)
    for rep in range(REPS - 1):
        opt.reset(clear_info=True)
        opt.vectorizer.vector[:] = jnp.array(
            np.random.randn(opt.d) * np.pi, dtype=JAX_DTYPE
        )
        opt.set_loss_var(jnp.asarray(ind, dtype=jnp.int64))
        opt.optimize(N_ITER)
        tmp_energies.append(opt.loss)
    best_idx = np.argmin(tmp_energies)
    ev = tmp_energies[best_idx]
    energies.append(ev)
# %%
plot_circuit(hierq)