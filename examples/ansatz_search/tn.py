# %%
import os
import time
import numpy as np
import sympy as sp
from collections import namedtuple
from functools import reduce
from utils import *
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
)
import logging
import sys
from utils import *


import quimb as qu
import quimb.tensor as qtn
class AbstractMPS:
    """
    Abstract connected tensor, basically a gate which has an arity for physical indices, in, out but a virutal bond aswell
    These are the units that gets repeated in a motif, also depends on parameters
    indices of tensors provided should be auxilary, physical out, physical in, such that the gate is obtained by contracting over the aux index
    information flows from bottom to top if thinking about a circuit and from left to right (altough I might want to change this)
    We assume at the moment that there's only on physical in physical out direction
    Arity must be greater than 1, in the list of tensors their shapes can either be of length 3 i.e. 3 for ones on edges and 4 in between or all 4 if "periodic"
    This is basically just an MPS

    """

    def __init__(self, tensors, n_symbols, symbols=None, name=None, hierq=None):
        self.arity = len(tensors)
        self.tensors = tensors
        self.n_symbols = n_symbols
        self.symbols = symbols
        self.value = self()
        self.edge=None
        self.name = name
        self.hierq = hierq

    def __call__(self, symbols=None, store=False):
        if symbols is None:
            if self.symbols is None:
                return None
            else:
                symbols = self.symbols
        N = self.arity
        if len(self.tensors[0](symbols).shape) == 3:
            tmp = Qinit(range(1, N)) + Qcycle(boundary="open")
            auxcons = tmp[1].E
            connections = (
                [(1, -1, -1 - N)]
                + [
                    con + (-k, -(k + N))
                    for con, k in zip(auxcons, range(2, len(auxcons) + 2))
                ]
                + [(N - 1, -(N), -2 * N)]
            )
        else:
            tmp = Qinit(range(1, N + 1)) + Qcycle(boundary="periodic")
            auxcons = tmp[1].E
            connections = [
                con + (-k, -(k + N - 1))
                for con, k in zip(auxcons, range(1, len(auxcons) + 1))
            ]
        value = ncon([tensor(symbols) for tensor in self.tensors], connections)
        if store is True:
            self.value = value
        return value
    def get_symbols(self):
        """
        Get symbols for this unitary.

        Returns: List of symbols
        """
        return self.symbols

    def set_symbols(self, symbols=None):
        """
        Set symbols for this unitary.

        Args:
            symbols (list): List of symbols
        """

        if len(symbols) != self.n_symbols:
            raise ValueError(
                f"Number of symbols must be {self.n_symbols} for this function"
            )
        self.symbols = symbols

    def set_edge(self, edge):
        self.edge = edge



# %%
θs = [0.3,.5,.7,.9,1.1,1.3]
data = np.array([[0,1],[1,0]])
inds = ('i0', 'i1')
tags = ('X',)
# tzy= qtn.Tensor(ZYe(θs[0]))
N=15
hierq = Qinit(N)+Qcycle(mapping=qYe)+Qcycle(mapping=qcry)+Qcycle(stride=2,mapping=qcry)+Qcycle(mapping=qZYe)+Qcycle(stride=2,mapping=qYZe)+Qcycle(mapping=qYe)
hierq.set_symbols(θs)

# circ = hierq(backend="quimb")
# print(np.dot(ψ.conj(), Zi(N,3)@Zi(N,4)@ψ))

# circ.psi.draw()
# %%
# circ.local_expectation(qu.pauli('Z') & qu.pauli('Z'), (43, 49))
# %%

#%%


# %%
# from quimb.tensor import *
# qmps=circ.psi
# circ.psi.draw()
# MPO_origin=MPO_ham_heis(L=N, j=(1.0,1.0,1.0), bz=0.0, S=0.5, cyclic=False)
# MPO_origin=MPO_origin.astype('complex128')
# psi_h=qmps.H 
# qmps.align_(MPO_origin, psi_h)
# print ("E_init", ( psi_h & MPO_origin & qmps).contract(all, optimize='auto-hq').real, ( qmps.H & qmps).contract(all, optimize='auto-hq'))


# %%
# tnopt_qmps=auto_diff_gate(qmps, MPO_origin, GATE, optimizer_c='L-BFGS-B')
#  #tnopt_qmps=auto_diff_stateGATE(qmps, p_DMRG, optimizer_c='L-BFGS-B')


# tnopt_qmps.optimizer = 'L-BFGS-B' 
# qmps = tnopt_qmps.optimize( n=400, ftol= 2.220e-10, maxfun= 10e+9, gtol= 1e-12, eps= 1.49016e-08, maxls=400, iprint = 5, disp=True )

# %%
import numpy as np
import quimb as qu
import quimb.tensor as qtn
# import parital
from functools import partial
from numpy import exp as e, cos as c, sin as s, array as arr,outer, sqrt, kron, diag
from autoray import astype, backend_like, do, get_dtype_name, reshape
import jax
import jax.numpy as jnp
from hierarqcal import *
import tensorflow as tf

def O0O1e(O0,O1):
    def generic_unitary(θs):
        θ = θs[0]
        with backend_like(θ):
            zero = θ * 0.0
            c = do("complex", do("cos", θ), zero)
            s = do("complex", zero,  do("sin", θ))
            I4 = do("array", qu.identity(4), like=θ, dtype=c.dtype)
            O0O1 = do("array",
                    qu.kron(qu.pauli(O0), qu.pauli(O1)),
                    like=θ, dtype=c.dtype)
            U = c * I4 + s * O0O1
            return do("reshape", U, (2, 2, 2, 2))
    return generic_unitary
def Oe(O0):
    def generic_unitary(θs):
        θ = θs[0]
        with backend_like(θ):
            zero = θ * 0.0
            c = do("complex", do("cos", θ), zero)
            s = do("complex", zero,  do("sin", θ))
            I2 = do("array", qu.identity(2), like=θ, dtype=c.dtype)
            U0 = do("array",qu.pauli(O0),
                    like=θ, dtype=c.dtype)
            U = c * I2 + s * U0
            return do("reshape", U, (2, 2))
    return generic_unitary


for lbl in ["XX", "YY", "ZZ", "XY", "XZ", "YX", "YZ", "ZX", "ZY"]:
    globals()[f"{lbl}e"] = O0O1e(lbl[0],lbl[1])
for lbl in ["X", "Y", "Z"]:
    globals()[f"{lbl}e"] = Oe(lbl)
    

    


θs = [jnp.array([.3]),jnp.array([.5])]
quYe = qtn.array_ops.PArray(Ye, θs[0])
quZYe = qtn.array_ops.PArray(XZe, θs[1])
circ = qtn.Circuit(2)
circ.apply_gate_raw(quYe, [0], tags='RY')

circ.apply_gate_raw(quYe, [1], tags='RY')
circ.apply_gate_raw(quZYe, [0,1])
# circ.set_params(θs)
# circ.draw(color=['PSI0',"A"])
circ.psi.draw(color=['PSI0',"A"])
# %%
for lbl in ["XX", "YY", "ZZ", "XY", "XZ", "YX", "YZ", "ZX", "ZY"]:
    globals()[f"qu{lbl}e"] = Qunitary(get_quimb_as_f(O0O1e(lbl[0],lbl[1])), n_symbols=1,arity=2)
for lbl in ["X", "Y", "Z"]:
    globals()[f"qu{lbl}e"] = Qunitary(get_quimb_as_f(Oe(lbl)),n_symbols=1,arity=1)
# %%
N=15
hierq = Qinit(N)+Qcycle(mapping=quYe)+Qcycle(mapping=quZYe)+Qcycle(stride=2,mapping=quZYe)+Qcycle(mapping=quXYe)+Qcycle(stride=2,mapping=quYXe)+Qcycle(mapping=quYe)
param_vals = [0.3, 0.5, 0.7, 0.9,1.1,1.3]
params_tf = [tf.Variable([val], dtype=tf.float64, name=f"param_{i}") for i, val in enumerate(param_vals)]
hierq.set_symbols(params_tf)
circ = hierq(backend="quimb")
# %%
# circ = qtn.Circuit(5)
# θs = [jnp.array([.3]),jnp.array([.5])]
# quYe = qtn.array_ops.PArray(Ye, θs[0])
# quZYe = qtn.array_ops.PArray(XZe, θs[1])
# circ = qtn.Circuit(2)
# circ.apply_gate(quYe, 0)
# circ.apply_gate(quYe, 1)
# circ.apply_gate(quZYe, 0,1)
# circ.set_params(θs)
# %%
from quimb.tensor import *
qmps=circ.psi
circ.psi.draw()
MPO_origin=MPO_ham_heis(L=N, j=(1.0,1.0,1.0), bz=0.0, S=0.5, cyclic=False)
MPO_origin=MPO_origin.astype('complex128')
psi_h=qmps.H 
qmps.align_(MPO_origin, psi_h)
print ("E_init", ( psi_h & MPO_origin & qmps).contract(all, optimize='auto-hq'), ( qmps.H & qmps).contract(all, optimize='auto-hq'))
# %%
qmps.draw(color=['PSI0'])
# %%
import autoray
def energy_gate(qmps, MPO):
   psi_h=qmps.H 
   qmps.align_(MPO, psi_h)
   E_complex=(( psi_h & MPO & qmps).contract(all, optimize='auto-hq'))
   return  autoray.do('real',  E_complex)


def auto_diff_gate(qmps,MPO, optimizer_c='L-BFGS-B'):

 tnopt_qmps= qtn.TNOptimizer(
    qmps,                      
    loss_fn=energy_gate,                    
    loss_constants={ "MPO": MPO},
    constant_tags=['PSI0'], 
    tags=[parm.name for parm in  params_tf], 
    shared_tags=[parm.name for parm in  params_tf],
    autodiff_backend="tensorflow",   # use 'autograd' for non-compiled optimization
    optimizer='L-BFGS-B',     # the optimization algorithm
)
 return tnopt_qmps
# %%
tmp = auto_diff_gate(qmps, MPO_origin, optimizer_c='L-BFGS-B')
result = tmp.optimize(10)
# %%
result.get_params()
# %%
tnopt_qmps = auto_diff_gate(qmps, MPO_origin, optimizer_c='L-BFGS-B')
tnopt_qmps.optimizer = 'L-BFGS-B' 
qmps = tnopt_qmps.optimize( n=50, ftol= 2.220e-10, maxfun= 10e+9, gtol= 1e-12, eps= 1.49016e-08, maxls=400, iprint = 5, disp=True )
#%%
print(circ.amplitude('0000001'))
# print(circ.amplitude('01'))
# print(circ.amplitude('10'))
# print(circ.amplitude('11'))
# %%

# %%
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
)
from utils import *
θsnp = [.3,.5,.7,.9]
hierq = Qinit(tensors = [ket0]*N)+Qcycle(mapping=qYe)+Qcycle(mapping=qZYe)+Qcycle(stride=2,mapping=qZYe)+Qcycle(mapping=qYe)
plot_circuit(hierq)
hierq.set_symbols(θsnp)
ψ = hierq().reshape(-1)
print(ψ)
# %%
print()
# %%
# -- ②  REGISTER WITH QUIMB ---------------------------------------------
qtn.circuit.register_param_gate("CRZ_CUSTOM", crz_param_gen, num_qubits=2)

# (optional) nice Python method so you can write qc.crz_custom(...)
def crz_custom(self, theta, ctrl, targ, *, parametrize=False, **kw):
    self.apply_gate("CRZ_CUSTOM", theta, ctrl, targ,
                    parametrize=parametrize, **kw)
setattr(qtn.Circuit, "crz_custom", crz_custom)


# -- ③  USE IT -----------------------------------------------------------
qc = qtn.Circuit(2)

# a) concrete gate – θ is a number --------------------------------------
theta = 0.3
qc.crz_custom(theta, 0, 1)

# b) symbolic / differentiable gate -------------------------------------
theta_sym = np.array([.8])        # could be a jax/cupy array, torch tensor …
qc.crz_custom(theta_sym, 0, 1, parametrize=True)

# now qc has two CRZ_CUSTOM tensors: the first is fixed, the second is
# still storing 'theta_sym' and can be updated later:
print("learnable parameters:", qc.get_params())

# update them in place
qc.set_params({1: np.array([.8])})

# at any point you can evaluate amplitudes, expectations, gradients, etc.
print("|ψ〉 size:", qc.to_dense().shape)



class AbstractMPS:
    """
    Abstract connected tensor, basically a gate which has an arity for physical indices, in, out but a virutal bond aswell
    These are the units that gets repeated in a motif, also depends on parameters
    indices of tensors provided should be auxilary, physical out, physical in, such that the gate is obtained by contracting over the aux index
    information flows from bottom to top if thinking about a circuit and from left to right (altough I might want to change this)
    We assume at the moment that there's only on physical in physical out direction
    Arity must be greater than 1, in the list of tensors their shapes can either be of length 3 i.e. 3 for ones on edges and 4 in between or all 4 if "periodic"
    This is basically just an MPS

    """

    def __init__(self, tensors, n_symbols, symbols=None, name=None, hierq=None):
        self.arity = len(tensors)
        self.tensors = tensors
        if callable(self.tensors[0]):
            self.as_function = True
        else:
            self.as_function = False
        self.n_symbols = n_symbols
        self.symbols = symbols
        self.value = self()
        self.edge=None
        self.name = name
        self.hierq = hierq

    def __call__(self, symbols=None, store=False):
        if symbols is None:
            if self.symbols is None:
                return None
            else:
                symbols = self.symbols
        N = self.arity
        if self.as_function is True:
            tensor_values = [tensor(symbols) for tensor in self.tensors]
        else:
            tensor_values = self.tensors
        if len(tensor_values[0].shape) == 3:
            tmp = Qinit(range(1, N)) + Qcycle(boundary="open")
            auxcons = tmp[1].E
            connections = (
                [(1, -1, -1 - N)]
                + [
                    con + (-k, -(k + N))
                    for con, k in zip(auxcons, range(2, len(auxcons) + 2))
                ]
                + [(N - 1, -(N), -2 * N)]
            )
        else:
            tmp = Qinit(range(1, N + 1)) + Qcycle(boundary="periodic")
            auxcons = tmp[1].E
            connections = [
                con + (-k, -(k + N - 1))
                for con, k in zip(auxcons, range(1, len(auxcons) + 1))
            ]
        value = ncon(tensor_values, connections)
        if store is True:
            self.value = value
        return value
    def get_symbols(self):
        """
        Get symbols for this unitary.

        Returns: List of symbols
        """
        return self.symbols

    def set_symbols(self, symbols=None):
        """
        Set symbols for this unitary.

        Args:
            symbols (list): List of symbols
        """

        if len(symbols) != self.n_symbols:
            raise ValueError(
                f"Number of symbols must be {self.n_symbols} for this function"
            )
        self.symbols = symbols

    def set_edge(self, edge):
        self.edge = edge