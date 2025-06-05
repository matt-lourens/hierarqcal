# %%
import numpy as np
import quimb as qu
from autoray import astype, backend_like, do, get_dtype_name, reshape
from hierarqcal import *
import time
import autoray
import torch
import tensorflow as tf
import jax, jax.numpy as jnp, quimb.tensor as qtn
from quimb import tree_map


def O0O1e(O0, O1):
    def generic_unitary(θs):
        θ = θs[0]
        with backend_like(θ):
            zero = θ * 0.0
            c = do("complex", do("cos", θ), zero)
            s = do("complex", zero, do("sin", θ))
            I4 = do("array", qu.identity(4), like=θ, dtype=c.dtype)
            pauli0 = do("array", qu.pauli(O0), like=θ, dtype=c.dtype)
            pauli1 = do("array", qu.pauli(O1), like=θ, dtype=c.dtype)
            O0O1 = do("kron", pauli0, pauli1, like=θ)
            # O0O1 = do(
            #     "array", qu.kron(qu.pauli(O0), qu.pauli(O1)), like=θ, dtype=c.dtype
            # )
            U = c * I4 + s * O0O1
            return do("reshape", U, (2, 2, 2, 2))

    return generic_unitary


def Oe(O0):
    def generic_unitary(θs):
        θ = θs[0]
        with backend_like(θ):
            zero = θ * 0.0
            c = do("complex", do("cos", θ / 2), zero)
            s = do("complex", zero, do("sin", θ / 2))
            I2 = do("array", qu.identity(2), like=θ, dtype=c.dtype)
            U0 = do("array", qu.pauli(O0), like=θ, dtype=c.dtype)
            U = c * I2 - s * U0
            return do("reshape", U, (2, 2))

    return generic_unitary


def COe(O0):
    """
    # ──────────────────────────────────────────────────────────────────────
    # Controlled exponential of a single-qubit Pauli operator σ_O.
    # Returns a function generic_unitary(θs) with shape (2,2,2,2) suitable
    # for wrapping in  Qunitary(…, n_symbols=1, arity=2).
    # ──────────────────────────────────────────────────────────────────────
    """
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


MAPPING_DICT ={}
for lbl in ["XX", "YY", "ZZ", "XY", "XZ", "YX", "YZ", "ZX", "ZY"]:
    globals()[f"e{lbl}"] = Qunitary(
        get_quimb_as_f(O0O1e(lbl[0], lbl[1])), n_symbols=1, arity=2, name=f"e{lbl}"
    )
    MAPPING_DICT[f"e{lbl}"] = globals()[f"e{lbl}"]
for lbl in ["X", "Y", "Z"]:
    globals()[f"e{lbl}"] = Qunitary(
        get_quimb_as_f(Oe(lbl)), n_symbols=1, arity=1, name=f"e{lbl}"
    )
    MAPPING_DICT[f"e{lbl}"] = globals()[f"e{lbl}"]
for lbl in ["X", "Y", "Z"]:
    globals()[f"cr{lbl}"] = Qunitary(
        get_quimb_as_f(COe(lbl)), n_symbols=1, arity=2, name=f"cr{lbl}"
    )
    MAPPING_DICT[f"cr{lbl}"] = globals()[f"cr{lbl}"]
# %%
N = 5
sub = Qinit(3, name="cycleXY") + Qcycle(mapping=eXY, boundary="periodic")
motif = (
    Qcycle(mapping=eY, boundary="open")
    + Qcycle(mapping=crY, boundary="open")
    + Qcycle(stride=2, mapping=crY, boundary="open")
    + Qcycle(mapping=eYZ, boundary="open")
    + Qcycle(stride=2, mapping=eYZ, boundary="open")
    + Qcycle(mapping=sub, boundary="open")
    + Qcycle(mapping=eY, boundary="open")
)
hierq = Qinit(N, state=qtn.Circuit(N)) + motif
# param_vals = np.random.rand(hierq.n_symbols).astype(np.float64)
param_vals = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
params_tf = [
    {"name": f"x{i}", "val": jnp.array([val], dtype=jnp.float32)}
    for i, val in enumerate(param_vals)
]
# TODO make more than 32 if needed double precision
hierq.set_symbols(params_tf)
circ = hierq(backend="quimb")
psi = circ.psi
H_RANGE = np.linspace(0, 1.0, 20)
ALL_MPOs = {}
size = N
for ind, hv in enumerate(H_RANGE):
    H0, Hb = get_ising_mpo_pbc(1, hv)
    mpo = qtn.MatrixProductOperator([H0] + [Hb] * (size - 1))
    # mpov, mpo_skeleton = qtn.pack(mpo)
    mpo = mpo.astype("complex128")
    ALL_MPOs[(size, ind)] = mpo
# %%
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


def energy_gate(psi, hind):
    mpo = make_mpo(hind)  # ← ordinary MPO object
    psi_h = psi.H
    psi.align_(mpo, psi_h)
    E_cplx = (psi_h & mpo & psi).contract(all, optimize="auto-hq")
    return autoray.do("real", E_cplx)


def auto_diff_gate(qmps, optimizer_c="L-BFGS-B"):

    tnopt_qmps = qtn.TNOptimizer(
        qmps,
        loss_fn=energy_gate,
        # loss_constants={"MPO": MPO},
        tags=[parm["name"] for parm in params_tf],
        shared_tags=[parm["name"] for parm in params_tf],
        autodiff_backend="jax",  # use 'autograd' for non-compiled optimization
        optimizer="L-BFGS-B",  # the optimization algorithm
    )
    return tnopt_qmps


# %%
opt = auto_diff_gate(psi, optimizer_c="L-BFGS-B")
for hind, hv in enumerate(H_RANGE):
    opt.set_loss_var(jnp.asarray(hind, dtype=jnp.int32))
    result = opt.optimize(10)
    # print(opt.loss)

# %%
# from copy import copy, deepcopy

# def motif_to_dict(motif):
#     motif = deepcopy(motif) 
#     motif_dict = vars(motif)
#     if motif_dict["mapping"] is None:
#         return motif_dict
#     else: 
#         mapping_dict = vars(motif_dict["mapping"])
#     if mapping_dict["hierq"] is None:
#         if mapping_dict.get("function", None):            
#             del mapping_dict["function"]
#         motif_dict["mapping"] = mapping_dict
#         return motif_dict
#     else:
#         hierq = mapping_dict["hierq"]
#         motif_dict["mapping"] ={}
#         ind = 0
#         current = hierq.tail
#         # motif_dict["mapping"][ind] = vars(current)
#         while current is not None:
#             motif_dict["mapping"][ind] = motif_to_dict(current)
#             current = current.next
#             ind += 1
            
#         return motif_dict

# def motifs_to_dict(motif):
#     if not(isinstance(motif,Qmotifs)):
#         motif = Qmotifs(motif)
        
    # motif_dict = {}
    # for ind, m in enumerate(motif):
    #     motif_dict[ind] = motifs_to_dict(m)
    # return motif_dict
motif_dict = motifs_to_dict(motif)

# %%
# import inspect
# from hierarqcal import PRIMITIVE_CLASS_MAP
# # reconstruct

# def dict_to_motifs(motif_dict):
#     motif_dict_cp = motif_dict.copy()
#     new_motif = Qmotifs()
#     for ind, m in motif_dict_cp.items():
#         cls = PRIMITIVE_CLASS_MAP[m["type"]]
#         del m["type"]
#         mapping_dict = m["mapping"]
#         if mapping_dict is None:
#             pass
#         elif mapping_dict.get(0,None) is None:
#             mapping =  globals()[mapping_dict["name"]]
#             m["mapping"] = mapping
#         else:
#             m["mapping"] = dict_to_motifs(mapping_dict)
#         if cls == Qinit:
#             del m["is_operation"]
#             del m["Q_avail"]
#             new_motif = Qinit(**m)
#         else:
#             motif = cls(**m)
#             new_motif = new_motif + motif
#     return new_motif
    
tmp = dict_to_motifs(motif_dict,MAPPING_DICT)
# print("hi")   
# %%
# test
hierq = Qinit(N, state=qtn.Circuit(N)) + tmp
# param_vals = np.random.rand(hierq.n_symbols).astype(np.float64)
param_vals = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
params_tf = [
    {"name": f"x{i}", "val": jnp.array([val], dtype=jnp.float32)}
    for i, val in enumerate(param_vals)
]
# TODO make more than 32 if needed double precision
hierq.set_symbols(params_tf)
circ = hierq(backend="quimb")
psi = circ.psi
H_RANGE = np.linspace(0, 1.0, 20)
ALL_MPOs = {}
size = N
for ind, hv in enumerate(H_RANGE):
    H0, Hb = get_ising_mpo_pbc(1, hv)
    mpo = qtn.MatrixProductOperator([H0] + [Hb] * (size - 1))
    # mpov, mpo_skeleton = qtn.pack(mpo)
    mpo = mpo.astype("complex128")
    ALL_MPOs[(size, ind)] = mpo
# %%
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


def energy_gate(psi, hind):
    mpo = make_mpo(hind)  # ← ordinary MPO object
    psi_h = psi.H
    psi.align_(mpo, psi_h)
    E_cplx = (psi_h & mpo & psi).contract(all, optimize="auto-hq")
    return autoray.do("real", E_cplx)


def auto_diff_gate(qmps, optimizer_c="L-BFGS-B"):

    tnopt_qmps = qtn.TNOptimizer(
        qmps,
        loss_fn=energy_gate,
        # loss_constants={"MPO": MPO},
        tags=[parm["name"] for parm in params_tf],
        shared_tags=[parm["name"] for parm in params_tf],
        autodiff_backend="jax",  # use 'autograd' for non-compiled optimization
        optimizer="L-BFGS-B",  # the optimization algorithm
    )
    return tnopt_qmps


# %%
opt = auto_diff_gate(psi, optimizer_c="L-BFGS-B")
for hind, hv in enumerate(H_RANGE):
    opt.set_loss_var(jnp.asarray(hind, dtype=jnp.int32))
    result = opt.optimize(10)
    # print(opt.loss)
# print("hey")
# %%
get_global_pattern = lambda: np.random.choice(
    ["*1", "1*", "*1*", "1*1", "*1*1", "1*1*", "*1*1*", "1*1*1"]
    + ["!0", "0!", "!0!", "0!0", "!0!0", "0!0!", "!0!0!", "0!0!0"]
    + ["*!", "!*", "!*!", "*!*", "01", "10", "101", "010", "1001", "0110"]
)
get_merge_within_pattern = lambda: np.random.choice(["*1", "1*"])

motifs = Qcycle(mapping=eXY)+Qmask("*!")+Qcycle(mapping=eY, share_weights=False)+Qunmask("previous")+Qcycle(mapping=eX)
hierq = Qinit(N, state=qtn.Circuit(N))+motifs
param_vals = np.random.rand(hierq.n_symbols)
params_tf = [
    {"name": f"x{i}", "val": jnp.array([val], dtype=jnp.float32)}
    for i, val in enumerate(param_vals)
]
hierq.set_symbols(params_tf)
circ = hierq(backend="quimb")
circ.draw()
# %%
tmp = motifs_to_dict(motifs)
# %%
tmp = dict_to_motifs(tmp,MAPPING_DICT)
hierq = Qinit(N, state=qtn.Circuit(N))+tmp
param_vals = np.random.rand(hierq.n_symbols)
params_tf = [
    {"name": f"x{i}", "val": jnp.array([val], dtype=jnp.float32)}
    for i, val in enumerate(param_vals)
]
hierq.set_symbols(params_tf)
circ = hierq(backend="quimb")
circ.draw()
print("oi")