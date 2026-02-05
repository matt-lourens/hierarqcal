# %%
import os

os.environ["JAX_ENABLE_X64"] = "True"
import jax

jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp

JAX_DTYPE = jnp.float64
import numpy as np
import quimb as qu
from hierarqcal import *
import jax, jax.numpy as jnp
import quimb.tensor as qtn
from autoray import astype, backend_like, do, get_dtype_name, reshape
from functools import reduce
from itertools import product
from itertools import combinations

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


# def sumOs(Os, share_symbols=False):
#     def generic_unitary(θs):
#         θ = θs[0]
#         if share_symbols is True:
#             θs = [θ] * len(Os)
#         with backend_like(θ):
#             zero = θ * 0.0
#             one = zero + 1.0  # Ensure it's a tensor type
#             img = do("complex", zero, one)
#             I = do("array", qu.identity(2), like=θ, dtype=img.dtype)
#             terms = []
#             for ind, O in enumerate(Os):
#                 oplist = [I] * len(Os)
#                 oplist[ind] = do("array", 1 / 2 * qu.pauli(O), like=θ, dtype=img.dtype)
#                 term = reduce(lambda A, B: do("kron", A, B, like=θ), oplist)
#                 terms.append(θs[ind] * term)
#             U = do("linalg.expm", -img * sum(terms))  # sum of terms
#             return do("reshape", U, (2,) * (2 * len(Os)))

#     return generic_unitary

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
            nq = len(Os[0]) # assuming all are same length
            for ind1, Ostring in enumerate(Os):
                oplist = [I] * len(Ostring)
                for ind2, O in enumerate(Ostring):
                    oplist[ind2] = do("array", 1 / 2 * qu.pauli(O), like=θ, dtype=img.dtype)
                term = reduce(lambda A, B: do("kron", A, B, like=θ), oplist)
                terms.append(θs[ind1] * term)
            U = do("linalg.expm", -img * sum(terms))
            return do("reshape", U, (2,) * (2 * nq))

    return generic_unitary

GATES = [
    "eXX",
    "eYY",
    "eZZ",
    "eXY",
    "eXZ",
    "eYX",
    "eYZ",
    "eZX",
    "eZY",
    "eI",
    "eX",
    "eY",
    "eZ",
]
ALL_GATES = True
MAPPING_DICT = {}
MAX_PAULISTRING_LEN = 3
pauli_strings = [
    "".join(pauli_string)
    for reps in range(1, MAX_PAULISTRING_LEN + 1)
    for pauli_string in product(["X", "Y", "Z"], repeat=reps)
]
one_length = [ps for ps in pauli_strings if len(ps) == 1]
two_length = [ps for ps in pauli_strings if len(ps) == 2]
three_length = [ps for ps in pauli_strings if len(ps) == 3]

one_combos = [
    combo
    for reps in range(2,3)
    for combo in combinations(one_length, r=reps)
]

two_combos = [
    combo
    for reps in range(2,3)
    for combo in combinations(two_length, r=reps)
]
# one_combos = product(one_length, repeat=2)
#%%
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
        get_quimb_as_f(sumOs(Os, share_symbols=False)), n_symbols=len(Os), arity=len(Os[0]), name=varname
    )
    MAPPING_DICT[varname] = globals()[varname]
    if varname in GATES or ALL_GATES:
        hierq_gates.append(globals()[varname])
    
for Os in two_combos:
    varname = "".join(["e"] + ["p".join([lbl for lbl in Os])])
    globals()[varname] = Qunitary(
        get_quimb_as_f(sumOs(Os, share_symbols=False)), n_symbols=len(Os), arity=len(Os[0]), name=varname
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
        
    
# if len(lbl) > 1:
#     # sums that don't share symbol
#     varname = "".join(["e"] + [f"{char}p" for char in lbl])[:-1:]
#     globals()[varname] = Qunitary(
#         get_quimb_as_f(sumOs(lbl, share_symbols=False)),
#         n_symbols=len(lbl),
#         arity=len(lbl),
#         name=varname,
#     )
#     MAPPING_DICT[varname] = globals()[varname]
#     if varname in GATES or ALL_GATES:
#         hierq_gates.append(globals()[varname])
#     # sums that share symbol
#     varname = "".join(["se"] + [f"{char}p" for char in lbl])[:-1:]
#     globals()[varname] = Qunitary(
#         get_quimb_as_f(sumOs(lbl, share_symbols=True)),
#         n_symbols=1,
#         arity=len(lbl),
#         name=varname,
#     )
#     MAPPING_DICT[varname] = globals()[varname]
#     if varname in GATES or ALL_GATES:
#         hierq_gates.append(globals()[varname])


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


# %%
N = 11
H_RANGE = np.linspace(0, 1, 20)
ALL_MPOs = {}
for ind, hv in enumerate(H_RANGE):
    H0, Hb = get_ising_mpo_pbc(1, hv)
    mpo = qtn.MatrixProductOperator([H0] + [Hb] * (N - 1))
    mpo = mpo.astype("complex128")
    ALL_MPOs[(N, ind)] = mpo
raw_bank = []
for hind, h in enumerate(H_RANGE):
    raw, _ = qtn.pack(ALL_MPOs[(N, hind)])
    raw_bank.append({k: jnp.asarray(v) for k, v in raw.items()})
keys = sorted(raw_bank[0].keys())
stacked = {k: jnp.stack([rb[k] for rb in raw_bank]) for k in keys}
_, mpo_skeleton = qtn.pack(ALL_MPOs[(N, 0)])


def make_mpo(hind):  # hind is 0-D int32 JAX array
    raw = {k: stacked[k][hind] for k in keys}  # numeric pytree
    return qtn.unpack(raw, mpo_skeleton)


def energy(psi, hind):
    mpo = make_mpo(hind)  # ← ordinary MPO object
    psi_h = psi.H
    psi.align_(mpo, psi_h)
    E_cplx = (psi_h & mpo & psi).contract(all, optimize="auto-hq")
    return do("real", E_cplx)


def get_optimiser(qmps, optimizer_c="L-BFGS-B", params=[]):

    tnopt_qmps = qtn.TNOptimizer(
        qmps,
        loss_fn=energy,
        # loss_constants={"MPO": MPO},
        tags=[parm["name"] for parm in params],
        shared_tags=[parm["name"] for parm in params],
        autodiff_backend="jax",  # use 'autograd' for non-compiled optimization
        optimizer="L-BFGS-B",  # the optimization algorithm
        progbar=True,
    )
    return tnopt_qmps


# %%
# sub = (
#     Qinit(2, name="?")
#     + Qmotif(E=[(1,)], mapping=eY)
#     + Qcycle(mapping=eYZ, boundary="open")
#     + Qmotif(E=[(1,)], mapping=eY)
#     + Qmotif(E=[(1,)], mapping=eI)
# )
# motif = Qinit(5)+Qcycle(mapping=eY, boundary="open")#+ Qcycle(    mapping=eZpYpZ, boundary="periodic")+Qpivot("1*", mapping=sub)
# motif1 = Qcycle(mapping=eY)
hierq = Qinit(N, state=qtn.Circuit(N)) +Qcycle(stride=1,mapping=eXY) +Qcycle(mapping=eY)  +Qcycle(mapping=eZY) # Qcycle(stride=3,mapping=eYZ)+ Qcycle(mapping=eZY)
param_vals = np.random.rand(hierq.n_symbols)
# param_vals = [0.3, 0.5]#, 0.7, 0.9,1.1,1.3]

params = [
    {"name": f"x{i}", "val": jnp.array(val, dtype=jnp.float64)}
    for i, val in enumerate(param_vals)
]
hierq.set_symbols(params)
circ = hierq(backend="quimb")
psi = circ.psi
opt = get_optimiser(psi, optimizer_c="L-BFGS-B", params=params)
# circ.draw()
# %%
motfs_dict = Qmotifs((motif1,)).to_dict()
a = Qmotifs.from_dict(motfs_dict, MAPPING_DICT)
# %%
psi = circ.psi
opt = get_optimiser(psi, optimizer_c="L-BFGS-B", params=params)
energies = []
for hind, hv in enumerate(H_RANGE):
    opt.set_loss_var(jnp.asarray(hind, dtype=jnp.int64))
    result = opt.optimize(10)
    Ev = opt.loss / N
    print(f"h:{hv}, E:{Ev}")
    energies.append(Ev)
print(np.mean(energies))
# %%
N_ITER=100
REPS=5
energies = []
pathinds = [hind for hind, hv in enumerate(H_RANGE)]
for ind in pathinds:
    tmp_energies = []
    opt.set_loss_var(jnp.asarray(ind, dtype=jnp.int64))
    _ = opt.optimize(N_ITER)
    tmp_energies.append(opt.loss)
    for rep in range(REPS-1):
        opt.reset(clear_info=True)
        opt.vectorizer.vector[:] = jnp.array(np.random.randn(opt.d)*np.pi, dtype=JAX_DTYPE)
        opt.set_loss_var(jnp.asarray(ind, dtype=jnp.int64))
        opt.optimize(N_ITER)
        tmp_energies.append(opt.loss)
    best_idx = np.argmin(tmp_energies)
    ev = tmp_energies[best_idx]
    energies.append(ev)
aes = np.array(energies, dtype=JAX_DTYPE)/N
# %%
import matplotlib.pyplot as plt

# diff = np.abs(aesZY) - np.abs(aesXY)

diff = np.abs(aesZY) - np.abs(aesXY)

fig, ax = plt.subplots(figsize=(10, 6))
ax.axhline(0, color='k', lw=0.8)                 # reference

# colour each point: red = aesZY > aesXY, blue = aesXY > aesZY
colors = np.where(diff >= 0, 'crimson', 'royalblue')
ax.scatter(H_RANGE, diff, c=colors, s=30)

ax.set_yscale('symlog', linthresh=1e-6)          # keeps zeros + sign
ax.set_xlabel('H')
ax.set_ylabel('|aesZY| – |aesXY|  (symlog)')
ax.set_title('Signed energy difference (colour shows sign)')


# %%
N_ITER=100
REPS=5
energies = []
tmp_energies = []
opt.set_loss_var(jnp.asarray(10, dtype=jnp.int64))
opt.vectorizer.vector[:] = jnp.array(np.random.randn(opt.d)*np.pi, dtype=JAX_DTYPE)
_ = opt.optimize(N_ITER)
ev1 = opt.loss/N
print(ev1)
# energies.append(ev)
# aes = np.array(energies, dtype=JAX_DTYPE)/N
# %%
N_ITER=100
REPS=5
energies = []
tmp_energies = []
opt.set_loss_var(jnp.asarray(10, dtype=jnp.int64))
opt.vectorizer.vector[:] = jnp.array(np.random.randn(opt.d)*np.pi, dtype=JAX_DTYPE)
_ = opt.optimize(N_ITER)
ev2 = opt.loss/N
print(ev2)
# %%
ψ0 = np.zeros(2**N, dtype=np.complex128)
for ind, n in enumerate(range(2**N)):
    ψ0[ind] = circ.amplitude(f"{n:05b}")
# %%
hv = H_RANGE[10]
H = qtn.MPO_ham_ising(N,-1,hv,cyclic=True)
dmrg = qtn.DMRG2(H , bond_dims=[2], cutoffs=1e-10)
dmrg.solve(tol=1e-6, verbosity=1)
# %%
a = dmrg.state
# %%

# %%
import scipy.linalg as la
from utils import *

eY = lambda θ: la.expm(-1j * θ / 2 * (Y))
eZY = lambda θ: la.expm(-1j * θ/4 * (np.kron(Z, Y)))
eZZZ = lambda θ: la.expm(-1j * θ/8 * (np.kron(Z, kron(Z,Z))))
eYZ = lambda θ: la.expm(1j * θ * (np.kron(Y, Z)))
eXpY = lambda θ0, θ1: la.expm(
    -1j * (θ0 * 1 / 2 * X + θ1 * 1 / 2 * Y)
)
eZXpZZ = lambda θ0, θ1: la.expm(
    -1j * (θ0 * 1 / 4 * np.kron(Z, X) + θ1 * 1 / 4 * np.kron(Z, Z) )
)
eZpYpX = lambda θ0, θ1, θ2: la.expm(
    -1j
    * (
        θ0 * 1 / 2 * kron(Z, kron(I, I))
        + θ1 * 1 / 2 * np.kron(I, kron(Y, I))
        + θ2 * 1 / 2 * np.kron(I, kron(I, X))
    )
)
seZpYpX = lambda θ0: la.expm(
    -1j
    * (
        θ0 * 1 / 2 * kron(Z, kron(I, I))
        + θ0 * 1 / 2 * np.kron(I, kron(Z, I))
        + θ0 * 1 / 2 * np.kron(I, kron(I, Z))
    )
)
esZpY = lambda θ0: la.expm(-1j * 1 / 2 * θ0 * (np.kron(Z, I) + np.kron(I, Y)))
qeXpY = Qunitary(get_tensor_as_f(eXpY), n_symbols=2, arity=1, name="qeXpY")
qeZXpZZ = Qunitary(get_tensor_as_f(eZXpZZ), n_symbols=2, arity=2, name="qeZXpZZ")
qseZpY = Qunitary(get_tensor_as_f(esZpY), n_symbols=1, arity=2, name="qseZpY")
qeZY = Qunitary(get_tensor_as_f(eZY), n_symbols=1, arity=2, name="qeZY")
qeYZ = Qunitary(get_tensor_as_f(eYZ), n_symbols=1, arity=2, name="qeYZ")
qeZZZ= Qunitary(get_tensor_as_f(eZZZ), n_symbols=1, arity=3, name="qeZZZ")
qeY = Qunitary(get_tensor_as_f(eY), n_symbols=1, arity=1, name="qeY")
qseZpYpX = Qunitary(get_tensor_as_f(seZpYpX), n_symbols=1, arity=3, name="qseZpYpX")
motif = Qcycle(mapping=qeZXpZZ)
hierq = Qinit(tensors=[ket0] * N) + motif
param_vals = [0.3, 0.5]
hierq.set_symbols(param_vals)
ψ1 = hierq().reshape(-1)
# %%
plot_circuit(hierq)
