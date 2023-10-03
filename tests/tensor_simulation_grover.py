from hierarqcal import (
    Qcycle,
    Qmotif,
    Qinit,
    Qmask,
    Qunmask,
    Qpermute,
    Qpivot,
    plot_circuit,
    plot_motif,
    get_tensor_as_f,
    Qunitary,
)
import numpy as np
import sympy as sp
import itertools as it


def get_probabilities(psi, basis_vectors=(np.array([1, 0]), np.array([0, 1]))):
    # We assume each basis vector is of the same length
    t_range = len(basis_vectors[0])
    # Calculate number of qubits/qutrits/.. etc
    nq = int(np.log(len(psi)) / np.log(t_range))
    # All possible amplitudes
    a = [{i for i in range(t_range)}] * nq
    bitstrings = list(it.product(*a))
    probs = dict()
    for bitstring in bitstrings:
        test = [basis_vectors[bit] for bit in bitstring]
        t = test[0]
        for tn in test[1:]:
            t = np.kron(t, tn)
        # Measurement operator
        M = np.outer(t, t)
        probs[bitstring] = np.abs(np.conj(psi).T @ np.conj(M).T @ M @ psi)
    return probs


H_m = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
X_m = np.array([[0, 1], [1, 0]])
CN_m = sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
x0 = sp.symbols("x")
CP_m = sp.Matrix(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, sp.exp(sp.I * x0)]]
)

H = Qunitary(get_tensor_as_f(H_m), 0, 1)
X = Qunitary(get_tensor_as_f(X_m), 0, 1)
CP = Qunitary(get_tensor_as_f(CP_m), arity=2, symbols=[x0])
CN = Qunitary(get_tensor_as_f(CN_m), 0, 2)


h_bottom = Qpivot("*1", mapping=H)
phase_pivot = Qpivot("*1", merge_within="01", mapping=CP(-np.pi / 2))
cnot_phase_cnot = (
    Qmotif(E=[(0, 1)], mapping=CN)
    + Qmotif(E=[(1, 2)], mapping=CP(np.pi / 2))
    + Qmotif(E=[(0, 1)], mapping=CN)
)


toffoli = (
    Qinit([i for i in range(3)]) + h_bottom + phase_pivot + cnot_phase_cnot + h_bottom
)

# Unitary to prepare the state |psi>
U_psi = Qcycle(mapping=H)
# Unitary to prepare the target state |T>
# Mask ancillary qubits
ancilla_str = "00100"
maskAncillas = Qmask(ancilla_str)
unmask_ancillas = Qunmask("previous")
# Multicontrolled Z gate
multiCZ = h_bottom
multiCZ += Qcycle(mapping=toffoli, step=2, boundary="open")
multiCZ += Qmask("*1")
multiCZ += Qcycle(mapping=toffoli, step=2, boundary="open", edge_order=[-1])
multiCZ += Qunmask("previous")
multiCZ += Qpivot(mapping=H, global_pattern="*1")

# Reflections in the plane orthogonal to |0>: I - 2|0><0|
U_reflect_0 = (
    Qcycle(mapping=X) + unmask_ancillas + multiCZ + maskAncillas + Qcycle(mapping=X)
)

# Oracle
U_oracle = U_reflect_0
# Reflection in the hyperplane orthogonal to |psi> (also called the defusion operator)
U_defuse = U_psi + U_reflect_0 + U_psi

# Grover operator
grover = U_oracle + U_defuse

tensors = [np.array([1, 0], dtype=np.complex256)] * 5
# Initialise the circuit and prepare the initial state |psi>
groverCircuit = (
    Qinit([i for i in range(5)], tensors=tensors) + maskAncillas + U_psi + grover
)

psi = groverCircuit()
# =======
n_controls = 5
n_work = 4

# Number of controls + number of work + 1 target qubit
nq = n_controls + n_work + 1

single_toffoli = Qmotif(E=[(0, 1, n_controls)], mapping=Qunitary(None, arity=3))
cycle_toffolis = (
    Qmask("11*1")
    + Qpivot(
        "*!1",
        merge_within="011",
        mapping=Qunitary(None, arity=3),
        boundaries=("open", "open", "open"),
    )
    + Qunmask("previous")
)
single_control_u = Qmotif(E=[(nq - 2, nq - 1)], mapping=Qunitary(None, arity=2))

u = Qinit(nq) + single_toffoli + cycle_toffolis + single_control_u
plot_circuit(u)
print("hi")
