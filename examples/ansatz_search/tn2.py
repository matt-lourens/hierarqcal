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
            O0O1 = do(
                "array", qu.kron(qu.pauli(O0), qu.pauli(O1)), like=θ, dtype=c.dtype
            )
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
            U = qu.kron(P0_b, I2_b) + qu.kron(P1_b, U_t)

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




var_gen = tf.Variable([.3], dtype=tf.float64)
for lbl in ["XX", "YY", "ZZ", "XY", "XZ", "YX", "YZ", "ZX", "ZY"]:
    globals()[f"qu{lbl}e"] = qtn.array_ops.PArray(O0O1e(lbl[0],lbl[1]),var_gen)
for lbl in ["X", "Y", "Z"]:
    globals()[f"qu{lbl}e"] = qtn.array_ops.PArray(Oe(lbl),var_gen)
for lbl in ["X", "Y", "Z"]:
    globals()[f"qucr{lbl}"] = qtn.array_ops.PArray(COe(lbl),var_gen)

# %%

# %%
ket0 = tf.Variable([1,0], dtype=tf.float64)
ket1 = tf.Variable([0,1], dtype=tf.float64)
ψ00 = qtn.Tensor(data=ket0, inds=["k3"], tags=qtn.oset(['hello', 'world']))
ψ01 = qtn.Tensor(data=ket0, inds=["k4"], tags=qtn.oset(['hello', 'world']))
opYZ = qtn.PTensor(fn=O0O1e("Z","Y"), params=var_gen, inds=[1,2,3,4], tags="A")
# mps = qtn.MatrixProductState([ψ00,ψ01],sites=["k0","k1"], L=2)
# %%
# mps.draw(["A", "PSI0"])
# mps.contract().data
# %%
# %%
N=5
ket0 = np.array([1,0])
mps = qtn.MPS_product_state(tuple([ket0]*N))
mps.add_tensor(opYZ)
# %%
mps.draw()
# %%
# mps.contract(all, optimize="auto-hq").data
k, h = 0.3, 0.5
H0, Hb, HN = get_H(k, h)
MPO= qtn.MatrixProductOperator([H0] + [Hb] * (N - 2) + [HN])
MPO= MPO.astype("complex128")
# %%
mpsh = mps.H
a,b,c = mps.align_(MPO, mpsh)
(a | b | c).contract(all, optimize="auto-hq")
# %% verify
# from hierarqcal import *
# from utils import *
# hierq = Qinit(tensors=[ket0,ket0])+Qcycle(mapping=qZYe)
# hierq.set_symbols([.3])
# hierq()
# %%
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
N = 2
psi0 = qtn.MPS_computational_state('0' * N)
psi0.draw()
# %%
from utils import *
def bd(θs):
    θ = θs[0]
    with backend_like(θ):
        zero = θ * 0.0
        c = do("complex", do("cos", θ/2), zero)
        s = do("complex", zero, do("sin", θ/2))
        w = do("array", [c*I, s*Y], like=θ, dtype=c.dtype)
        return do("reshape", w, (2, 2,2))

b = qtn.Tensor(qtn.array_ops.PArray(bd,var_gen),inds=["z0","k0","z1"])
# %%
psi0=psi0&b
# %%
Rz = qu.phase_gate(0.42)
psi0.gate(Rz,0,contract="swap+split")