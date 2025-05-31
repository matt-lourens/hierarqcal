import os
import numpy as np
from numpy import exp as e, cos as c, sin as s, array as arr,outer, sqrt, kron, diag
from functools import reduce, partial
import sympy as sp
from hierarqcal import Qunitary, get_tensor_as_f
import scipy.linalg as la
from typing import List, Union, Tuple, Optional

# Helpers for constructing tensor products
def Opi(op, L, i):
    return reduce(np.kron, [np.eye(2) if i != k else op for k in range(L)])

def Opall(op, L):
    return reduce(np.kron, [op for _ in range(L)])
PATH = os.path.dirname(os.path.dirname(__file__))
i = 1j
# Define Pauli matrices and Identity
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

# Define all two-qubit Pauli products
XX = np.kron(X, X)
YY = np.kron(Y, Y)
ZZ = np.kron(Z, Z)
XY = np.kron(X, Y)
XZ = np.kron(X, Z)
YX = np.kron(Y, X)
YZ = np.kron(Y, Z)
ZX = np.kron(Z, X)
ZY = np.kron(Z, Y)

# Parameterized two-qubit entanglers for each combination
XXe = lambda θ: la.expm(1j * θ * XX)
YYe = lambda θ: la.expm(1j * θ * YY)
ZZe = lambda θ: la.expm(1j * θ * ZZ)
XYe = lambda θ: la.expm(1j * θ * XY)
XZe = lambda θ: la.expm(1j * θ * XZ)
YXe = lambda θ: la.expm(1j * θ * YX)
YZe = lambda θ: la.expm(1j * θ * YZ)
ZXe = lambda θ: la.expm(1j * θ * ZX)
ZYe = lambda θ: la.expm(1j * θ * ZY)
Ye = lambda θ: la.expm(-1j * θ * Y)

# Parameterized partial-SWAP gate
def p_swap(theta):
    swap = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    return la.expm(1j * theta * swap)

# Define fixed two-qubit gates
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
])
sqrt_SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0.5*(1+1j), 0.5*(1-1j), 0],
    [0, 0.5*(1-1j), 0.5*(1+1j), 0],
    [0, 0, 0, 1]
])
iSWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1j, 0],
    [0, 1j, 0, 0],
    [0, 0, 0, 1]
])

# Other single-qubit gates and helpers
H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
Ii, Xi, Yi, Zi = partial(Opi, I), partial(Opi, X), partial(Opi, Y), partial(Opi, Z)
R = lambda O, θ: c(θ/2)*I - 1j * s(θ/2)*O
ket0 = np.array([1, 0])
ket1 = np.array([0, 1])
Uz2x = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
Uz2y = 1 / np.sqrt(2) * np.array([[1, -1j], [1, 1j]])
H_annni = lambda N, k, h: (
    -1/4 * reduce(np.add, [Zi(N, i) @ Zi(N, (i + 1) % N) for i in range(N)])
    + k/4 * reduce(np.add, [Zi(N, i) @ Zi(N, (i + 2) % N) for i in range(N)])
    - h/2 * reduce(np.add, [Xi(N, i) for i in range(N)])
)
H_annni_norm = lambda N, k, h: (
    -reduce(np.add, [Zi(N, i) @ Zi(N, (i + 1) % N) for i in range(N)])
    + k*reduce(np.add, [Zi(N, i) @ Zi(N, (i + 2) % N) for i in range(N)])
    - h*reduce(np.add, [Xi(N, i) for i in range(N)])
)
H_ising= lambda N=3, h=.5: (
    -1/4 * reduce(np.add, [Zi(N, i) @ Zi(N, (i + 1) % N) for i in range(N)])
    - h/2 * reduce(np.add, [Xi(N, i) for i in range(N)])
)
rx = partial(R, X)
ry = partial(R, Y)
rz = partial(R, Z)
crx = lambda θ: np.kron(np.outer(ket0, ket0), I) + np.kron(np.outer(ket1, ket1), rx(θ))
cry = lambda θ: np.kron(np.outer(ket0, ket0), I) + np.kron(np.outer(ket1, ket1), ry(θ))
crz = lambda θ: np.kron(np.outer(ket0, ket0), I) + np.kron(np.outer(ket1, ket1), rz(θ))

# Three-qubit entanglers
# (a) A symmetric three-body entangler: exp[iθ(X⊗X⊗X + Y⊗Y⊗Y + Z⊗Z⊗Z)]
XXX = np.kron(np.kron(X, X), X)
YYY = np.kron(np.kron(Y, Y), Y)
ZZZ = np.kron(np.kron(Z, Z), Z)
three_body_op = XXX + YYY + ZZZ
three_ent = lambda θ: la.expm(1j * θ * three_body_op)


# Path handling (if needed)
PATH = os.path.dirname(os.path.dirname(__file__))

# Wrap gates into Qunitary objects
qh      = Qunitary(get_tensor_as_f(H), arity=1, n_symbols=0, name="h")
qx      = Qunitary(get_tensor_as_f(X), arity=1, n_symbols=0, name="x")
qy      = Qunitary(get_tensor_as_f(Y), arity=1, n_symbols=0, name="y")
qz      = Qunitary(get_tensor_as_f(Z), arity=1, n_symbols=0, name="z")
qrx     = Qunitary(get_tensor_as_f(rx), n_symbols=1, arity=1, name="rx")
qry     = Qunitary(get_tensor_as_f(ry), n_symbols=1, arity=1, name="ry")
qrz     = Qunitary(get_tensor_as_f(rz), n_symbols=1, arity=1, name="rz")
qcrx    = Qunitary(get_tensor_as_f(crx), n_symbols=1, arity=2, name="crx")
qcry    = Qunitary(get_tensor_as_f(cry), n_symbols=1, arity=2, name="cry")
qcrz    = Qunitary(get_tensor_as_f(crz), n_symbols=1, arity=2, name="crz")
qXXe    = Qunitary(get_tensor_as_f(XXe), n_symbols=1, arity=2, name="XXe")
qYYe    = Qunitary(get_tensor_as_f(YYe), n_symbols=1, arity=2, name="YYe")
qZZe    = Qunitary(get_tensor_as_f(ZZe), n_symbols=1, arity=2, name="ZZe")
qXYe    = Qunitary(get_tensor_as_f(XYe), n_symbols=1, arity=2, name="XYe")
qXZe    = Qunitary(get_tensor_as_f(XZe), n_symbols=1, arity=2, name="XZe")
qYXe    = Qunitary(get_tensor_as_f(YXe), n_symbols=1, arity=2, name="YXe")
qYZe    = Qunitary(get_tensor_as_f(YZe), n_symbols=1, arity=2, name="YZe")
qZXe    = Qunitary(get_tensor_as_f(ZXe), n_symbols=1, arity=2, name="ZXe")
qZYe    = Qunitary(get_tensor_as_f(ZYe), n_symbols=1, arity=2, name="ZYe")
qYe    = Qunitary(get_tensor_as_f(Ye), n_symbols=1, arity=1, name="Ye")
qCNOT   = Qunitary(get_tensor_as_f(CNOT), arity=2, n_symbols=0, name="cnot")
qpswap  = Qunitary(get_tensor_as_f(p_swap), n_symbols=1, arity=2, name="pswap")
qiSWAP  = Qunitary(get_tensor_as_f(iSWAP), arity=2, n_symbols=0, name="iswap")
qSqrtSwap = Qunitary(get_tensor_as_f(sqrt_SWAP), arity=2, n_symbols=0, name="sqrt_swap")
q3ent   = Qunitary(get_tensor_as_f(three_ent), n_symbols=1, arity=3, name="3ent")

# Final hierarchical gate list (all in one line)



def ncon(tensors: List[np.ndarray],
         connects: List[Union[List[int], Tuple[int]]],
         con_order: Optional[Union[List[int], str]] = None,
         check_network: Optional[bool] = True,
         which_env: Optional[int] = 0):
  """
  Network CONtractor: contracts a tensor network of N tensors via a sequence
  of (N-1) tensordot operations. More detailed instructions and examples can
  be found at: https://arxiv.org/abs/1402.0939.
  Args:
    tensors: list of the tensors in the network.
    connects: length-N list of lists (or tuples) specifying the network
      connections. The jth entry of the ith list in connects labels the edge
      connected to the jth index of the ith tensor. Labels should be positive
      integers for internal indices and negative integers for free indices.
    con_order: optional argument to specify the order for contracting the
      positive indices. Defaults to ascending order if omitted. Can also be
      set at "greedy" or "full" to call a solver to automatically determine
      the order.
    check_network: if true then the input network is checked for consistency;
      this can catch many common user mistakes for defining networks.
    which_env: if provided, ncon will produce the environment of the requested
      tensor (i.e. the network given by removing the specified tensor from
      the original network). Only valid for networks with no open indices.
  Returns:
    Union[np.ndarray,float]: the result of the network contraction; an
      np.ndarray if the network contained open indices, otherwise a scalar.
  """
  num_tensors = len(tensors)
  tensor_list = [tensors[ele] for ele in range(num_tensors)]
  connect_list = [np.array(connects[ele]) for ele in range(num_tensors)]

  # generate contraction order if necessary
  flat_connect = np.concatenate(connect_list)
  if con_order is None:
    con_order = np.unique(flat_connect[flat_connect > 0])
  else:
    con_order = np.array(con_order)

  # check inputs if enabled
  if check_network:
    dims_list = [list(tensor.shape) for tensor in tensor_list]
    check_inputs(connect_list, flat_connect, dims_list, con_order)

  # do all partial traces
  for ele in range(len(tensor_list)):
    num_cont = len(connect_list[ele]) - len(np.unique(connect_list[ele]))
    if num_cont > 0:
      tensor_list[ele], connect_list[ele], cont_ind = partial_trace(
          tensor_list[ele], connect_list[ele])
      con_order = np.delete(
          con_order,
          np.intersect1d(con_order, cont_ind, return_indices=True)[1])

  # do all binary contractions
  while len(con_order) > 0:
    # identify tensors to be contracted
    cont_ind = con_order[0]
    locs = [
        ele for ele in range(len(connect_list))
        if sum(connect_list[ele] == cont_ind) > 0
    ]

    # do binary contraction
    cont_many, A_cont, B_cont = np.intersect1d(
        connect_list[locs[0]],
        connect_list[locs[1]],
        assume_unique=True,
        return_indices=True)
    if np.size(tensor_list[locs[0]]) < np.size(tensor_list[locs[1]]):
      ind_order = np.argsort(A_cont)
    else:
      ind_order = np.argsort(B_cont)

    tensor_list.append(
        np.tensordot(
            tensor_list[locs[0]],
            tensor_list[locs[1]],
            axes=(A_cont[ind_order], B_cont[ind_order])))
    connect_list.append(
        np.append(
            np.delete(connect_list[locs[0]], A_cont),
            np.delete(connect_list[locs[1]], B_cont)))

    # remove contracted tensors from list and update con_order
    del tensor_list[locs[1]]
    del tensor_list[locs[0]]
    del connect_list[locs[1]]
    del connect_list[locs[0]]
    con_order = np.delete(
        con_order,
        np.intersect1d(con_order, cont_many, return_indices=True)[1])

  # do all outer products
  while len(tensor_list) > 1:
    s1 = tensor_list[-2].shape
    s2 = tensor_list[-1].shape
    tensor_list[-2] = np.outer(tensor_list[-2].reshape(np.prod(s1)),
                               tensor_list[-1].reshape(np.prod(s2))).reshape(
                                   np.append(s1, s2))
    connect_list[-2] = np.append(connect_list[-2], connect_list[-1])
    del tensor_list[-1]
    del connect_list[-1]

  # do final permutation
  if len(connect_list[0]) > 0:
    return np.transpose(tensor_list[0], np.argsort(-connect_list[0]))
  else:
    return tensor_list[0].item()


def partial_trace(A, A_label):
  """ Partial trace on tensor A over repeated labels in A_label """

  num_cont = len(A_label) - len(np.unique(A_label))
  if num_cont > 0:
    dup_list = []
    for ele in np.unique(A_label):
      if sum(A_label == ele) > 1:
        dup_list.append([np.where(A_label == ele)[0]])

    cont_ind = np.array(dup_list).reshape(2 * num_cont, order='F')
    free_ind = np.delete(np.arange(len(A_label)), cont_ind)

    cont_dim = np.prod(np.array(A.shape)[cont_ind[:num_cont]])
    free_dim = np.array(A.shape)[free_ind]

    B_label = np.delete(A_label, cont_ind)
    cont_label = np.unique(A_label[cont_ind])
    B = np.zeros(np.prod(free_dim))
    A = A.transpose(np.append(free_ind, cont_ind)).reshape(
        np.prod(free_dim), cont_dim, cont_dim)
    for ip in range(cont_dim):
      B = B + A[:, ip, ip]

    return B.reshape(free_dim), B_label, cont_label

  else:
    return A, A_label, []


def check_inputs(connect_list, flat_connect, dims_list, con_order):
  """ Check consistancy of NCON inputs"""

  pos_ind = flat_connect[flat_connect > 0]
  neg_ind = flat_connect[flat_connect < 0]

  # check that lengths of lists match
  if len(dims_list) != len(connect_list):
    raise ValueError(
        ('mismatch between %i tensors given but %i index sublists given') %
        (len(dims_list), len(connect_list)))

  # check that tensors have the right number of indices
  for ele in range(len(dims_list)):
    if len(dims_list[ele]) != len(connect_list[ele]):
      raise ValueError((
          'number of indices does not match number of labels on tensor %i: '
          '%i-indices versus %i-labels')
          % (ele, len(dims_list[ele]), len(connect_list[ele])))

  # check that contraction order is valid
  if not np.array_equal(np.sort(con_order), np.unique(pos_ind)):
    raise ValueError(('NCON error: invalid contraction order'))

  # check that negative indices are valid
  for ind in np.arange(-1, -len(neg_ind) - 1, -1):
    if sum(neg_ind == ind) == 0:
      raise ValueError(('NCON error: no index labelled %i') % (ind))
    elif sum(neg_ind == ind) > 1:
      raise ValueError(('NCON error: more than one index labelled %i') % (ind))

  # check that positive indices are valid and contracted tensor dimensions match
  flat_dims = np.array([item for sublist in dims_list for item in sublist])
  for ind in np.unique(pos_ind):
    if sum(pos_ind == ind) == 1:
      raise ValueError(('NCON error: only one index labelled %i') % (ind))
    elif sum(pos_ind == ind) > 2:
      raise ValueError(
          ('NCON error: more than two indices labelled %i') % (ind))

    cont_dims = flat_dims[flat_connect == ind]
    if cont_dims[0] != cont_dims[1]:
      raise ValueError(
          ('NCON error: tensor dimension mismatch on index labelled %i: '
           'dim-%i versus dim-%i') % (ind, cont_dims[0], cont_dims[1]))

  return True

