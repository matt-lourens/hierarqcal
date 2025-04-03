import itertools as it
import sympy as sp
from scipy.optimize import minimize
import numpy as np
from utils import *
from collections import Counter


def objective_ising(symbols, motif, **Hkwargs):
    # np.dot(ψ.conj(), H(N,**Hkwargs) @ ψ).real
    h = Hkwargs.get("h", None)
    if h is None:
        raise ValueError("h must be provided as keyword arguments")
    N = len(motif.tail.Q)
    if symbols is not None:
        motif.set_symbols(symbols)
    ψ = motif()
    expZZ1 = 0
    expX1 = 0
    for j in range(N):
        zz1bits = (j, (j + 1) % N)
        x1bits = (j,)
        orig_shape = ψ.shape
        zz1nbits = tuple([j for j in range(len(orig_shape)) if j not in zz1bits])
        zz1nbits_size = int(np.prod([orig_shape[j] for j in zz1nbits]))
        zz1bits_size = int(np.prod([orig_shape[j] for j in zz1bits]))
        x1nbits = tuple([j for j in range(len(orig_shape)) if j not in x1bits])
        x1nbits_size = int(np.prod([orig_shape[j] for j in x1nbits]))
        x1bits_size = int(np.prod([orig_shape[j] for j in x1bits]))
        # put bits not acted on last
        zz1perm = zz1bits + zz1nbits
        x1perm = x1bits + x1nbits
        ψzz1 = ψ.transpose(zz1perm)
        ψx1 = ψ.transpose(x1perm)
        # turn into matrix
        ψzz1 = ψzz1.reshape(zz1bits_size, zz1nbits_size)
        ψx1 = ψx1.reshape(x1bits_size, x1nbits_size)
        expZZ1 += np.trace(ψzz1.conj().T @ np.kron(Z, Z) @ ψzz1)
        expX1 += h * np.trace(ψx1.conj().T @ X @ ψx1)
    # ψ = motif().reshape(-1)
    # print(np.dot(ψ.conj(), H(N,**Hkwargs) @ ψ).real - (expZZ1 + expZZ2 + expX1).real)
    return (-1 / 4 * expZZ1 + -h / 2 * expX1).real


def get_results(
    motif,
    H_params={sp.symbols("g"): np.arange(0.01, 2.2, 0.2)},
    N_iter=10,
    reps=1,
    verbose=False,
    objective="annni",
):
    if objective == "ising":
        objective = objective_ising
    else:
        raise ValueError("objective doesn't exist")
    results = []
    opt_symbols = {}
    Hpk = list(H_params.keys())
    if motif.n_symbols > 0:
        # symb_freq = Counter(motif.get_symbols())
        # n_symbols = len(symb_freq)
        n_symbols = motif.n_symbols
        symbols_tmp = np.array([0.1] * n_symbols, dtype=np.float64)
        for Hpv in list(it.product(*list(H_params.values()))):
            Hkwargs = {k: v for k, v in zip(Hpk, Hpv)}
            symbols_tmp, loss = optimize(
                motif,
                symbols_tmp,
                verbose=verbose,
                N_iter=N_iter,
                reps=reps,
                objective=objective,
                **Hkwargs,
            )
            results.append(loss)
            opt_symbols[Hpv] = symbols_tmp
    else:
        for Hpv in list(it.product(*list(H_params.values()))):
            Hkwargs = {k: v for k, v in zip(Hpk, Hpv)}
            results.append(objective(None, motif, **Hkwargs))
    return results


def optimize(
    motif, symbols, objective=None, verbose=False, N_iter=10, reps=1, **Hkwargs
):
    def func_to_minimize(symbols):
        return objective(symbols, motif, **Hkwargs)

    losses = []
    symbs = []
    for rep in range(reps):
        result = minimize(
            func_to_minimize,
            symbols,
            method="BFGS",
            options={"maxiter": N_iter, "disp": verbose},
            # method="L-BFGS-B"
        )
        symbols_opt = result.x
        loss = result.fun
        losses.append(loss)
        symbs.append(symbols_opt)
    ind = np.argmin(losses)
    θs = symbs[ind]
    loss = losses[ind]
    if verbose:
        N = len(motif.tail.Q)
        print(f"Optimization result for N={N}, loss={losses[ind]}")
    return θs, loss


# """
# Debug
# """
# from hierarqcal import Qinit, Qcycle
# from scipy.sparse.linalg import eigsh


# def numerical_diagonalization(H, *args):
#     return eigsh(H(*args), k=1, which="SA", return_eigenvectors=False)[0]


# N = 4
# m = (
#     Qinit(tensors=[ket0] * N)
#     + Qcycle(step=2, mapping=qry)
#     + Qcycle(step=2, offset=1, mapping=qry)
#     + Qcycle(step=2, offset=0, mapping=qcry)
#     + Qcycle(stride=1, step=2, offset=1, mapping=qcry)
#     + Qcycle(step=2, mapping=qry)
#     + Qcycle(step=2, offset=1, mapping=qry)
# )
# k, h = 1, 1.5
# print(objective([1, 1,1,1,1,1], m, H_annni, **{"k": k, "h": h}))
# ψ = m().reshape(-1)
# print(np.dot(ψ.conj(), H_annni(N, k=k, h=h) @ ψ).real)
# print("hey")
