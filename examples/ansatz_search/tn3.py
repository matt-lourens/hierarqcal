# %%
import numpy as np
import quimb as qu
import quimb.tensor as qtn
from autoray import astype, backend_like, do, get_dtype_name, reshape
from hierarqcal import *
import time
import autoray
import torch
import tensorflow as tf

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
            O0O1  = do("kron", pauli0, pauli1, like=θ)
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
            c = do("complex", do("cos", θ/2), zero)
            s = do("complex", zero, do("sin", θ/2))
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
            c = do("complex", do("cos", θ/2), zero)  #  cos θ
            s = do("complex", zero, do("sin", θ/2))  # i·sin θ

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

from quimb.tensor.circuit import register_param_gate

for lbl in ["XX", "YY", "ZZ", "XY", "XZ", "YX", "YZ", "ZX", "ZY"]:
    globals()[f"qu{lbl}e"] = Qunitary(
        get_quimb_as_f(f"qu{lbl}e".upper()), n_symbols=1, arity=2
    )
    register_param_gate(f"qu{lbl}e".upper(), O0O1e(lbl[0], lbl[1]), 2)
for lbl in ["X", "Y", "Z"]:
    globals()[f"qu{lbl}e"] = Qunitary(
        get_quimb_as_f(f"qu{lbl}e".upper()), n_symbols=1, arity=1
    )
    register_param_gate(f"qu{lbl}e".upper(), Oe(lbl), 1)
for lbl in ["X", "Y", "Z"]:
    globals()[f"qucr{lbl}"] = Qunitary(
        get_quimb_as_f(f"qucr{lbl}".upper()), n_symbols=1, arity=2
    )
    register_param_gate(f"qucr{lbl}".upper(), COe(lbl), 2)
# %%
# %%
import tensorflow as tf

"""
circuit mps
"""
N = 64
hierq = (
    Qinit(N, state=qtn.Circuit(N))
    + Qcycle(mapping=quYe, boundary="open")
    + Qcycle(mapping=qucrY, boundary="open")
    + Qcycle(stride=2, mapping=qucrY, boundary="open")
    + Qcycle(mapping=quYZe, boundary="open")
    + Qcycle(stride=2, mapping=quYZe, boundary="open")
    + Qcycle(mapping=quYe, boundary="open")
)
# param_vals = np.random.rand(hierq.n_symbols).astype(np.float64)
param_vals = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
params_tf = [
    {"name":f"x{i}","val":tf.Variable([val], dtype=tf.float64)}
    for i, val in enumerate(param_vals)
]
hierq.set_symbols(params_tf)
circ = hierq(backend="quimb")
# %%
psi = circ.psi
psi.draw(color = [parm.name for parm in params_tf])
# %%
k, h = 0.3, 0.5
psi = circ.psi
psi_h = psi.H
H0, Hb, HN = get_H(k, h)
MPO_origin = qtn.MatrixProductOperator([H0] + [Hb] * (N - 2) + [HN])
MPO_origin = MPO_origin.astype("complex128")
psi.align_(MPO_origin, psi_h)
# %%
t0 = time.time()
print(
    "E_init",
    (psi_h & MPO_origin & psi).contract(all, optimize="auto-hq")
)
t1 = time.time()
print(f"Time taken: {t1-t0:.3f} seconds")
print(type(psi))
# %%
def energy_gate(qmps, MPO):
    psi_h = qmps.H
    qmps.align_(MPO, psi_h)
    E_complex = (psi_h & MPO & qmps).contract(all, optimize="auto-hq")
    return autoray.do("real", E_complex)


def auto_diff_gate(qmps, MPO, optimizer_c="L-BFGS-B"):

    tnopt_qmps = qtn.TNOptimizer(
        qmps,
        loss_fn=energy_gate,
        loss_constants={"MPO": MPO},
        tags=[parm["name"] for parm in params_tf],
        shared_tags=[parm["name"] for parm in params_tf],
        autodiff_backend="tensorflow",  # use 'autograd' for non-compiled optimization
        optimizer="L-BFGS-B",  # the optimization algorithm
    )
    return tnopt_qmps


# %%
tmp = auto_diff_gate(psi, MPO_origin, optimizer_c="L-BFGS-B")
result = tmp.optimize(10)
# %%

# %%
import jax
from jax import numpy as jnp
# jax.config.update("jax_enable_x64", True)
# when searching we prob only nee single precision

"""
Now test out jax
"""
N = 64
hierq = (
    Qinit(N, state=qtn.Circuit(N))
    + Qcycle(mapping=quYe, boundary="open")
    + Qcycle(mapping=qucrY, boundary="open")
    + Qcycle(stride=2, mapping=qucrY, boundary="open")
    + Qcycle(mapping=quYZe, boundary="open")
    + Qcycle(stride=2, mapping=quYZe, boundary="open")
    + Qcycle(mapping=quYe, boundary="open")
)
# param_vals = np.random.rand(hierq.n_symbols).astype(np.float64)
param_vals = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
params_tf = [
    {"name":f"x{i}","val":jnp.array([val], dtype=jnp.float32)}
    for i, val in enumerate(param_vals)
]
# TODO make more than 32 if needed double precision
hierq.set_symbols(params_tf)
circ = hierq(backend="quimb")
# %%
# psi = circ.psi
# psi.draw(color = [parm["name"] for parm in params_tf])
# %%
k, h = 0.3, 0.5
psi = circ.psi
psi_h = psi.H
H0, Hb, HN = get_H(k, h)
MPO_origin = qtn.MatrixProductOperator([H0] + [Hb] * (N - 2) + [HN])
MPO_origin = MPO_origin.astype("complex128")
psi.align_(MPO_origin, psi_h)
# %%
t0 = time.time()
print(
    "E_init",
    (psi_h & MPO_origin & psi).contract(all, optimize="auto-hq")
)
t1 = time.time()
print(f"Time taken: {t1-t0:.3f} seconds")
print(type(psi))
# %%
def energy_gate(qmps, MPO):
    psi_h = qmps.H
    qmps.align_(MPO, psi_h)
    E_complex = (psi_h & MPO & qmps).contract(all, optimize="auto-hq")
    return autoray.do("real", E_complex)


def auto_diff_gate(qmps, MPO, optimizer_c="L-BFGS-B"):

    tnopt_qmps = qtn.TNOptimizer(
        qmps,
        loss_fn=energy_gate,
        loss_constants={"MPO": MPO},
        tags=[parm["name"] for parm in params_tf],
        shared_tags=[parm["name"] for parm in params_tf],
        autodiff_backend="jax",  # use 'autograd' for non-compiled optimization
        optimizer="L-BFGS-B",  # the optimization algorithm
    )
    return tnopt_qmps


# %%
tmp = auto_diff_gate(psi, MPO_origin, optimizer_c="L-BFGS-B")
result = tmp.optimize(10)
# %%
