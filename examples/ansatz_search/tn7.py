# %%
import numpy as np
import quimb as qu
from hierarqcal import *
import jax, jax.numpy as jnp
import quimb.tensor as qtn
from autoray import astype, backend_like, do, get_dtype_name, reshape
from functools import reduce
from itertools import product

jax.config.update("jax_enable_x64", True)


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
                G = θs[0]*reduce(lambda A, B: do("matmul", A, B, like=θ), terms)
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
max_string_length = 3
pauli_strings = [
    "".join(pauli_string)
    for reps in range(1, max_string_length + 1)
    for pauli_string in product(["I", "X", "Y", "Z"], repeat=reps)
]
for lbl in pauli_strings:
    varname = "".join(["e"] + [f"{char}" for char in lbl])
    globals()[varname] = Qunitary(
        get_quimb_as_f(prodOs(lbl)), n_symbols=1, arity=len(lbl), name=varname
    )
    MAPPING_DICT[varname] = globals()[varname]
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
        # sums that share symbol
        varname = "".join(["se"] + [f"{char}p" for char in lbl])[:-1:]
        globals()[varname] = Qunitary(
            get_quimb_as_f(sumOs(lbl, share_symbols=True)),
            n_symbols=1,
            arity=len(lbl),
            name=varname,
        )
        MAPPING_DICT[varname] = globals()[varname]

# Controlled rotations
for lbl in ["X", "Y", "Z"]:
    varname = f"cr{lbl}"
    globals()[varname] = Qunitary(
        get_quimb_as_f(COe(lbl)), n_symbols=1, arity=2, name=varname
    )
    MAPPING_DICT[varname] = globals()[varname]

def get_H(k_val, h_val):  # Renamed k,h to avoid conflict with module name if any
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    ZERO = np.zeros_like(I)
    Hb_local = np.array(
        [
            [I, ZERO, ZERO, ZERO],
            [Z, ZERO, ZERO, ZERO],
            [ZERO, I, ZERO, ZERO],
            [-h_val * X, -Z, k_val * Z, I],
        ]
    )  # lrud
    H0_local = np.array([-h_val * X, -Z, k_val * Z, I])
    HN_local = np.array([I, Z, ZERO, -h_val * X])

    # Hb_local = Hb_local.transpose(0,1,2,3)
    # H0_local = H0_local.transpose(0,1,2)
    # HN_local = HN_local.transpose(0,1, 2)
    return H0_local, Hb_local, HN_local
# %%
N = 6
Nk, Nh =1,1
H_RANGE = np.linspace(0.5, 0.7, Nh)
K_RANGE = np.linspace(0.3, 0.7, Nk)
ALL_MPOs = {}
inds =[]
for hind, hv in enumerate(H_RANGE):
    for kind, kv in enumerate(K_RANGE):
        H0, Hb, HN = get_H(kv, hv)
        mpo = qtn.MatrixProductOperator([H0] + [Hb] * (N - 2) + [HN])
        mpo = mpo.astype("complex128")
        ind = hind * Nk + kind
        inds.append(ind)
        ALL_MPOs[(N, ind)] = mpo
        
raw_bank = []
for ind in inds:
    raw, _ = qtn.pack(ALL_MPOs[(N, ind)])
    raw_bank.append({k: jnp.asarray(v) for k, v in raw.items()})
keys = sorted(raw_bank[0].keys())
stacked = {k: jnp.stack([rb[k] for rb in raw_bank]) for k in keys}
_, mpo_skeleton = qtn.pack(ALL_MPOs[(N, 0)])

# %%
motif = (Qcycle(mapping=eY, boundary="open")
    + Qcycle(mapping=crY, boundary="open")
    + Qcycle(stride=2, mapping=crY, boundary="open")
    + Qcycle(mapping=eYZ, boundary="open")
    + Qcycle(stride=2, mapping=eYZ, boundary="open")
    + Qcycle(mapping=eY, boundary="open"))
hierq = (
    Qinit(N, state=qtn.Circuit(N))+motif
    
)

# param_vals = np.random.rand(hierq.n_symbols)
param_vals = [.3,.5,.7,.9,1.1,1.3]
params = [
    {"name": f"x{i}", "val": jnp.array(val, dtype=jnp.float64)}
    for i, val in enumerate(param_vals)
]
hierq.set_symbols(params)
circ = hierq(backend="quimb")
psi =circ.psi
# %%
def make_mpo(ind):  # hind is 0-D int32 JAX array
    raw = {k: stacked[k][ind] for k in keys}  # numeric pytree
    return qtn.unpack(raw, mpo_skeleton)


def energy(psi, ind):
    mpo = make_mpo(ind)  # ← ordinary MPO object
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
energy(psi, 0)  # check initial energy
# %%
opt = get_optimiser(psi, optimizer_c="L-BFGS-B", params=params)
energies = []
for hind, hv in enumerate(H_RANGE):
    for kind, kv in enumerate(K_RANGE):
        ind = hind * Nk + kind
        opt.set_loss_var(jnp.asarray(ind, dtype=jnp.int64))
        result = opt.optimize(10)
        Ev = opt.loss / N
        print(f"h:{hv} k:{kv}, E:{Ev}")
        energies.append(Ev)
print(np.mean(energies))

# %%
energy(psi,0)
# %%
import time
k, h = 0.3, 0.5
qmps = circ.psi
# qmps = circMPS.psi
psi_h = qmps.H
H0, Hb, HN = get_H(k, h)
MPO_origin = qtn.MatrixProductOperator([H0] + [Hb] * (N - 2) + [HN])
MPO_origin = MPO_origin.astype("complex128")
qmps.align_(MPO_origin, psi_h)

t0 = time.time()
print(
    "E_init",
    (psi_h & MPO_origin & qmps).contract(all, optimize="auto-hq"),
    (qmps.H & qmps).contract(all, optimize="auto-hq"),
)
t1 = time.time()
print(f"Time taken: {t1-t0:.3f} seconds")

# %%
from utils import *
H_annni_norm = lambda N, k, h: (
    -reduce(np.add, [Zi(N, i) @ Zi(N, (i + 1) ) for i in range(N-1)])
    + k*reduce(np.add, [Zi(N, i) @ Zi(N, (i + 2) ) for i in range(N-2)])
    - h*reduce(np.add, [Xi(N, i) for i in range(N)])
)
motif = (Qcycle(mapping=qYe, boundary="open")
    + Qcycle(mapping=qcry, boundary="open")
    + Qcycle(stride=2, mapping=qcry, boundary="open")
    + Qcycle(mapping=qYZe, boundary="open")
    + Qcycle(stride=2, mapping=qYZe, boundary="open")
    + Qcycle(mapping=qYe, boundary="open"))
hierq = Qinit(tensors=[ket0]*N) + motif
hierq.set_symbols(param_vals)
ψ = hierq().reshape(-1)
en =np.conj(ψ) @(H_annni_norm(N, 0.3, 0.5) @ ψ)
print(en)