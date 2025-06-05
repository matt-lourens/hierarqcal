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
import jax
import jax.numpy as jnp

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

def get_Hannni(k_val, h_val): 
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

for lbl in ["XX", "YY", "ZZ", "XY", "XZ", "YX", "YZ", "ZX", "ZY"]:
    globals()[f"qu{lbl}e"] = Qunitary(
        get_quimb_as_f(O0O1e(lbl[0], lbl[1])), n_symbols=1, arity=2
    )
for lbl in ["X", "Y", "Z"]:
    globals()[f"qu{lbl}e"] = Qunitary(get_quimb_as_f(Oe(lbl)), n_symbols=1, arity=1)
for lbl in ["X", "Y", "Z"]:
    globals()[f"qucr{lbl}"] = Qunitary(get_quimb_as_f(COe(lbl)), n_symbols=1, arity=2)
# %%
N = 10
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
    {"name": f"x{i}", "val": jnp.array([val], dtype=jnp.float32)}
    for i, val in enumerate(param_vals)
]
# TODO make more than 32 if needed double precision
hierq.set_symbols(params_tf)
circ = hierq(backend="quimb")
# %%
psi = circ.psi
H_RANGE = np.linspace(0, 1.0, 20)
ALL_MPOs = {}
size=N
for ind,hv in enumerate(H_RANGE):
    H0, Hb = get_ising_mpo_pbc(1, hv)
    mpo = qtn.MatrixProductOperator([H0] + [Hb] * (size - 1))
    # mpov, mpo_skeleton = qtn.pack(mpo)
    mpo = mpo.astype("complex128")
    ALL_MPOs[(size, ind)] = mpo
    
# %%
# def energy_gate(psi, MPO):
#     psi_h = psi.H
#     psi.align_(MPO, psi_h)
#     E_complex = (psi_h & MPO & psi).contract(all, optimize="auto-hq")
#     return autoray.do("real", E_complex)


# def auto_diff_gate(qmps, MPO, optimizer_c="L-BFGS-B"):
#     tnopt_qmps = qtn.TNOptimizer(
#         qmps,
#         loss_fn=energy_gate,
#         loss_constants={"MPO": MPO},
#         tags=[parm["name"] for parm in params_tf],
#         shared_tags=[parm["name"] for parm in params_tf],
#         autodiff_backend="jax",  # use 'autograd' for non-compiled optimization
#         optimizer="L-BFGS-B",  # the optimization algorithm
#     )
#     return tnopt_qmps
# # %%
# # # energy_gate(psi, ALL_MPOs[(64,3)])
# # # %%
# # tmp = auto_diff_gate(psi, ALL_MPOs[(N,3)], optimizer_c="L-BFGS-B")
# # result = tmp.optimize(10)
# # %%
# def loss_fn(psi):
#     mpo = ALL_MPOs[(N,3)]
#     psi_h = psi.H
#     psi.align_(mpo, psi_h)
#     E_complex = (psi_h & mpo & psi).contract(all, optimize="auto-hq")
#     return autoray.do("real", E_complex)
# # %%
# loss_fn(psi,ALL_MPOs[(N,3)])
# # %%
# import jax
# import flax.linen as nn
# import optax

# def loss_fn(psi):
#     mpo = ALL_MPOs[(N,3)]
#     psi_h = psi.H
#     psi.align_(mpo, psi_h)
#     E_complex = (psi_h & mpo & psi).contract(all, optimize="auto-hq")
#     return autoray.do("real", E_complex)

# class CustomModule(nn.Module):
#     def setup(self):
#         params, skeleton = qtn.pack(psi)
#         self.skeleton = skeleton
#         self.params = {
#             i: self.param(f'param_{i}', lambda _: data)
#             for i, data in params.items()
#         }
    
#     def __call__(self):
#         psi = qtn.unpack(self.params, self.skeleton)
#         return loss_fn(psi,ALL_MPOs[(N,3)])

# model = CustomModule()
# params = model.init(jax.random.PRNGKey(42))
# loss_grad_fn = jax.value_and_grad(model.apply)
# tx = optax.adabelief(learning_rate=0.01)
# opt_state = tx.init(params)


# @jax.jit
# def step(params, opt_state):
#     loss, grads = loss_grad_fn(params)
#     updates, opt_state = tx.update(grads, opt_state, params)
#     params = optax.apply_updates(params, updates)
#     return params, opt_state, loss

# # # %%
# import tqdm

# its = 1_000
# pbar = tqdm.tqdm(range(its))

# for _ in pbar:
#     params, opt_state, loss_val = step(params, opt_state)
#     pbar.set_description(f"{loss_val}")

# # %%
# # minimal, loop-friendly tensor-network optimizer
import jax, jax.numpy as jnp
from functools import partial
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize

class SimpleTNOpt:
    def __init__(self, params, loss_fn, **loss_static):
        flat0, self.unravel = ravel_pytree(params)
        self._flat = jnp.asarray(flat0)              # warm-start vector
        self._vg = jax.jit(jax.value_and_grad(       # value+grad wrt params
            lambda flat, **kw: loss_fn(self.unravel(flat), **kw)
        ))
        self._loss_static = loss_static              # things that never change

    def step(self, maxiter=200, **loss_vars):
        args = {**self._loss_static, **loss_vars}
        def f(x):
            val, g = self._vg(x, **args)
            return float(val), jnp.array(g, float)
        out = minimize(f, self._flat, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter})
        self._flat = out.x                           # keep for next call
        return self.unravel(out.x), out.fun          # params*, loss

# ------- example usage -------------------------------------------------------
# toy loss: E(ψ; h) = ⟨ψ|H(h)|ψ⟩ with trivial “Hamiltonian”
def loss_fn(state, h):
    return jnp.vdot(state, state) + h * jnp.sum(state.real)

psi0 = jnp.ones(16) * 0.1
opt = SimpleTNOpt(psi0, loss_fn)                    # single construction

H_RANGE = jnp.linspace(-2.0, 2.0, 11)
energies = []
for h in H_RANGE:
    psi_opt, E = opt.step(h=h)                      # h passed via **loss_vars
    energies.append(E)

# %%

psi.get_params()