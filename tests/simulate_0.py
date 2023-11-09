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
    Qhierarchy,
)
import numpy as np
import sympy as sp
import itertools as it

def get_bitlist(a, n):
    """
    Returns a list of bits of length n representing the integer a.
    """
    a_bits = [int(b) for b in bin(a)[2:]]
    a_bits = [0] * (n - len(a_bits)) + a_bits
    return a_bits


def get_results(result_tensor, bit_layout):
    result = "".join([str(bit[0]) for bit in np.where(result_tensor == 1)])
    register_r = [bit for bit, label in zip(result, bit_layout) if label == "r"]
    register_x = [bit for bit, label in zip(result, bit_layout) if label == "x"]
    register_a = [bit for bit, label in zip(result, bit_layout) if label == "a"]
    register_b = [bit for bit, label in zip(result, bit_layout) if label == "b"]
    register_bc = [bit for bit, label in zip(result, bit_layout) if label == "bc"]
    register_c = [bit for bit, label in zip(result, bit_layout) if label == "c"]
    register_n = [bit for bit, label in zip(result, bit_layout) if label == "N"]
    register_n1 = [bit for bit, label in zip(result, bit_layout) if label == "t1"]
    print(f"r: {register_r}")
    print(f"x: {register_x}")
    print(f"a: {register_a}")
    print(f"b: {register_b} + {register_bc}")
    print(f"c: {register_c}")
    print(f"n: {register_n}")
    print(f"t: {register_n1}")
    return result, register_a, register_b, register_c, register_n, register_n1


def get_tensors(r, N, n, x=1, a=None, b=None):
    ket0 = np.array([1, 0])
    ket1 = np.array([0, 1])
    # Get bits
    a_bits = get_bitlist(a, n) if a is not None else [0] * n
    b_bits = get_bitlist(b, n) if b is not None else [0] * n
    modn = get_bitlist(N, n)
    r_bits = get_bitlist(r, n)
    x_bits = get_bitlist(x, n)
    # add r and x
    r_tensors = [ket0 if elem == 0 else ket1 for elem in r_bits[::-1]]
    x_tensors = [ket0 if elem == 0 else ket1 for elem in x_bits[::-1]]
    # init a,b,c to 0
    abc_tensors = [
        ket0 if elem == 0 else ket1
        for triplet in zip([0] * n, a_bits[::-1], b_bits[::-1])
        for elem in triplet
    ]
    bc_tensor = [ket0]
    N_tensors = [ket0 if elem == 0 else ket1 for elem in modn[::-1]]
    t0_tensor = [ket0]
    tensors = r_tensors + x_tensors + abc_tensors + bc_tensor + N_tensors + t0_tensor
    return tensors


# ====== Configuration

n = 3
N = 4
a = 6
r = 3
x = 1
b = 6
print(f"a - {a}\nx - {x}\nN - {N}\nr - {r}")
bit_layout = ["r"] * n + ["x"] * n + ["c", "a", "b"] * n + ["bc"] + ["N"] * n + ["t1"]
nq = len(bit_layout)
tensors = get_tensors(r, N, n, x=x, a=None, b=None)

mask_part = lambda part, inv=False: Qmask(
    "".join(
        [str((1 + inv) % 2) if b in part else str((0 + inv) % 2) for b in bit_layout]
    )
)

mask_part(["r","N"], inv=False)

# ====== Matrices
toffoli_m = np.identity(8)
toffoli_m[6, 6] = 0
toffoli_m[6, 7] = 1
toffoli_m[7, 6] = 1
toffoli_m[7, 7] = 0
cnot_m = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
h_m = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
swap_m = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
x_m = np.array([[0, 1], [1, 0]])
x0 = sp.symbols("x")
cp_m = sp.Matrix(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, sp.exp(sp.I * x0)]]
)


# ====== Gates level 0
toffoli = Qunitary(get_tensor_as_f(toffoli_m), 0, 3)
cnot = Qunitary(get_tensor_as_f(cnot_m), 0, 2)
hadamard = Qunitary(get_tensor_as_f(h_m), 0, 1)
swap = Qunitary(get_tensor_as_f(swap_m), 0, 2)
x = Qunitary(get_tensor_as_f(x_m), 0, 1)
cphase = Qunitary(get_tensor_as_f(cp_m), arity=2, symbols=[x0])

# ====== Motifs level 1
carry_motif = (
    Qinit(4)
    + Qmotif(E=[(1, 2, 3)], mapping=toffoli)
    + Qmotif(E=[(1, 2)], mapping=cnot)
    + Qmotif(E=[(0, 2, 3)], mapping=toffoli)
)

# Turn carry into a function to be able to reverse it easily
carry = lambda r: carry_motif if r == 1 else carry_motif.reverse()

sum = lambda r=1: Qinit(3) + Qpivot(
    "*1", merge_within="01", mapping=cnot, edge_order=[-1 * r]
)

# === verify
# plot_circuit(carry(1))
# plot_circuit(carry(-1))
# plot_circuit(sum(1))
# plot_circuit(sum(-1))
# hierq = (
#     Qinit(nq, tensors=tensors)
#     + Qmotif(E=[(0, 1, 2,3)], mapping=carry(1))
# )
# result_tensor = hierq()
# del(result_tensor)
# get_results(result_tensor, bit_layout)

# ====== Motifs level 2
carry_sum_motif = (
    lambda r=1: Qinit(4)
    + Qpivot("1*", merge_within="1111", mapping=carry(-r))
    + Qpivot("1*", merge_within="111", mapping=sum(r))
)
carry_sum = lambda r=1: carry_sum_motif(1) if r == 1 else carry_sum_motif(-1).reverse()


cnot_sum_motif = (
    lambda r=1: Qinit(3)
    + Qpivot("*1", merge_within="11", mapping=cnot)
    + Qpivot("*1", merge_within="111", mapping=sum(r))
)
cnot_sum = lambda r=1: cnot_sum_motif(1) if r == 1 else cnot_sum_motif(-1).reverse()
# === verify
# plot_circuit(carry_sum(1))
# plot_circuit(cnot_sum(-1))
# hierq = Qinit(nq, tensors=tensors) + Qmotif(E=[(0,1,2)],mapping=cnot_sum(-1))
# result_tensor = hierq()
# get_results(result_tensor, bit_layout)

# ====== Motifs level 3
carry_layer = lambda r=1: Qcycle(
    1,
    3,
    0,
    mapping=carry(r),
    boundary="open",
    edge_order=[r],
)
cnot_sum_pivot = lambda r: Qpivot("*10", merge_within="111", mapping=cnot_sum(r))
carry_sum_layer = lambda r: (
    Qmask("*111")
    + Qcycle(
        1,
        3,
        0,
        boundary="open",
        mapping=carry_sum(r),
        edge_order=[-r],
    )
    + Qunmask("previous")
)
addition = (
    mask_part(["r", "x", "N", "t1"], inv=False)
    + carry_layer(1)
    + cnot_sum_pivot(1)
    + carry_sum_layer(1)
    + Qunmask("previous")
)
subtraction = (
    mask_part(["r", "x", "N", "t1"], inv=False)
    + carry_sum_layer(-1)
    + cnot_sum_pivot(-1)
    + carry_layer(-1)
    + Qunmask("previous")
)

# === verify
# plot_circuit(Qinit(nq)+addition)
# plot_circuit(Qinit(nq)+subtraction)
# hierq = (
#     Qinit(nq, tensors=tensors)
#     + mask_part(["r", "x", "N", "t1"], inv=False)
#     + subtraction
# )
# result_tensor = hierq()
# get_results(result_tensor, bit_layout)
# del result_tensor
# plot_circuit(hierq)

# ====== Motifs level 4
"""
    Once we have an adder we can build one that does modulo N addition. Since because N is bigger than both a and b, their sum can't be larger than 2N. Therefore all we need to do is a and b and subtract N. If a+b is smaller than N we should not perform the subtraction, luckily this information is encoded in the b carry register (it's one if a+b>N so use that to decide whether we subtract N or not). We achieve this by swapping register a and N. Then we use t1 (temporary bit) to store whether a+b>n i.e. bc==1. So we flip bc, perform cnot onto t1 and flip back. Now we want to set all the ones of N to 0 if t1 is 1. This is done by masking all but register a and t1, then masking all the zeros of N and finally pivot cnots controls onto t1 where the targets cycle through the available bits of register a.
"""
# Build motif to swap a with N
swap_an = (
    mask_part(["a", "N"], inv=True)
    + Qcycle(n, 2, 0, mapping=swap, boundary="periodic")
    + Qunmask("previous")
)
# Build motif to flip temporary bit based on b carry
ctrl_not_bc_tmp_x = (
    mask_part(["bc", "t1"], inv=True)
    + Qpivot("10", mapping=x)
    + Qpivot("01", merge_within="01", mapping=cnot)
    + Qpivot("10", mapping=x)
    + Qunmask("previous")
)
# Build a motif that control flips the tempory bit without flipping bc
ctrl_not_bc_tmp = (
    mask_part(["bc", "t1"], inv=True)
    + Qpivot("01", merge_within="01", mapping=cnot)
    + Qunmask("previous")
)

# Build motif that sets "a" register to 0 if there was an overflow
zeros_of_N = "".join([str((bit + 1) % 2) for bit in get_bitlist(N, n)])
reset_if_overflow = (
    mask_part(["a", "t1"], inv=True)
    + Qmask(zeros_of_N)
    + Qpivot("*1", merge_within="10", mapping=cnot)
    + Qunmask("previous")
    + Qunmask("previous")
)

addition_mod_n = (

    addition
    + swap_an
    + subtraction
    + ctrl_not_bc_tmp_x
    + reset_if_overflow
    + addition
    + reset_if_overflow
    + swap_an
    + subtraction
    + ctrl_not_bc_tmp
    + addition
)

subtraction_mod_n = (
    Qinit(nq)
    + subtraction
    + ctrl_not_bc_tmp
    + addition
    + swap_an
    + reset_if_overflow
    + subtraction
    + reset_if_overflow
    + ctrl_not_bc_tmp_x
    + addition
    + swap_an
    + subtraction
)
# === verify
hierq = (
    Qinit(nq, tensors=tensors)
    + Qpivot(mapping=addition_mod_n) 
    + Qpivot(mapping=subtraction_mod_n)
)
# result_tensor = hierq()
# get_results(result_tensor, bit_layout)
# del result_tensor
# plot_circuit(hierq, plot_width=20)


# Multiplication level 5
def ctrl_mult(a, N, n, ctrl=0, divide=False):
    if divide is True:
        a_inv_mod = pow(a, -1, N)
        powers2a = [(a_inv_mod << k) % N for k in range(n - 1, -1, -1)]
    else:
        powers2a = [(a << k) % N for k in range(n - 1, -1, -1)]
    ctrl_mult_mod = tuple()
    for k in range(n):
        if powers2a[k] > 0:
            ctrl_r_string = "0" * ctrl + "1" + "0" * (n - 1 - ctrl)
            ctrl_x_string = "0" * (n - k - 1) + "1" + "*"
            bitstring = "".join(
                [str((bit + 1) % 2) for bit in get_bitlist(powers2a[k], n)]
            )
            mask_2apower = mask_part("a", True) + Qmask(bitstring[::-1])
            unmask_ctrl = Qunmask(ctrl_r_string + ctrl_x_string)
            encode_2a = (
                mask_2apower
                + unmask_ctrl
                + Qpivot("1*", merge_within="110", mapping=toffoli)
                + Qunmask("!")
            )
            ctrl_mult_mod += (
                encode_2a
                + Qpivot(mapping=addition_mod_n if not (divide) else subtraction_mod_n)
                + encode_2a
            )
            # ctrl_mult_mod_inv += encode_2a + Qpivot(mapping=subtraction_mod_n) + encode_2a
    copy_x_to_b = lambda r: (
        mask_part(["b", "x"], True)
        + Qunmask(ctrl_r_string + "*")
        + Qpivot("1*", mapping=x)
        + Qpivot(
            "1*", merge_within="100", mapping=toffoli, strides=[1, n, 0], edge_order=[r]
        )
        + Qpivot("1*", mapping=x)
        + Qunmask("!")
    )
    if divide:
        ctrl_mult_mod = Qinit(nq) + copy_x_to_b(-1) + ctrl_mult_mod
    else:
        ctrl_mult_mod = Qinit(nq) + ctrl_mult_mod + copy_x_to_b(1)
    return ctrl_mult_mod


# hierq = Qinit(nq, tensors=tensors) + ctrl_mult(a, N, n, ctrl=1, divide=False) + ctrl_mult(a, N, n, ctrl=1, divide=True)
# result_tensor = hierq()
# get_results(result_tensor, bit_layout)
# del result_tensor
# plot_circuit(hierq, plot_width=50)

# Exponentiation level 6
swap_bx = (
    mask_part(["b", "x"], True)
    + Qcycle(n, 1, 0, mapping=swap, boundary="open")
    + Qunmask("previous")
)
exp_mod_n = tuple()
for k in range(n):
    exp_mod_n += (
        Qpivot(mapping=ctrl_mult(a ** (2**k), N, n, ctrl=k, divide=False))
        + swap_bx
        + Qpivot(mapping=ctrl_mult(a ** (2**k), N, n, ctrl=k, divide=True))
    )

# level 7 qft
# qft = (
#     Qpivot(mapping=hadamard)
#     + Qpivot(
#         mapping=cphase,
#         share_weights=False,
#         symbol_fn=lambda x, ns, ne: np.pi * 2 ** (-ne),
#     )
#     + Qmask("1*")
# ) * n
hierq = Qinit(nq, tensors=tensors) + exp_mod_n #+ mask_part(["r"], inv=True) #+ qft

result_tensor = hierq()
# p = get_probabilities(result_tensor.flatten())
get_results(result_tensor, bit_layout)
del result_tensor
plot_circuit(hierq, plot_width=50)
print("hiou")
