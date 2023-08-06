import numpy as np
from hierarqcal import (
    Qhierarchy,
    Qcycle,
    Qpivot,
    Qpermute,
    Qmask,
    Qunmask,
    Qinit,
    Qmotif,
    Qmotifs,
    plot_motif,
    plot_circuit,
    Qunitary,
    Qsplit,
)


def binary_addition(x, y):
    # Convert the binary strings to integers
    x_int = int("".join(str(bit) for bit in x), 2)
    y_int = int("".join(str(bit) for bit in y), 2)

    # Add the integers
    sum_int = x_int + y_int

    # Convert the sum back to a binary string
    sum_bin = [
        int(bit) for bit in bin(sum_int)[2:]
    ]  # bin(sum_int) gives '0b...', so we skip the first two characters

    # Pad the sum with leading zeros if necessary to make it the same length as the inputs
    if len(sum_bin) < len(x):
        sum_bin = [0] * (len(x) - len(sum_bin)) + sum_bin

    return sum_bin


def half_adder(bits, symbols=None, state=None):
    b1, b2 = state[bits[0]], state[bits[1]]
    xor = b1 ^ b2
    carry = b1 and b2
    state[bits[0]] = carry
    state[bits[1]] = xor
    return state


def or_top(bits, symbols=None, state=None):
    b1, b2 = state[bits[0]], state[bits[1]]
    state[bits[0]] = b1 or b2
    return state


# input
x = [int(b) for b in bin(np.random.randint(0,512))[2:]]
y = [int(b) for b in bin(np.random.randint(0,512))[2:]]
bits = max(len(x),len(y))
n = 2 * bits
x = [0]*(bits - len(x)) + x
y = [0]*(bits - len(y)) + y

# program
full_adder = (
    Qinit(3)
    + Qcycle(mapping=Qunitary(half_adder, 0, 2), boundary="open")
    + Qpivot(global_pattern="1*", merge_within="11", mapping=Qunitary(or_top, 0, 2))
)
addition = (
    Qinit([i for i in range(n)], state=[elem for pair in zip(x, y) for elem in pair])
    + Qpivot("*1", "11", mapping=Qunitary(half_adder, 0, 2))
    + Qcycle(step=2, edge_order=[-1], mapping=full_adder, boundary="open")
)
# execute
b = addition()
pattern_fn = Qsplit.get_pattern_fn(
    None, pattern="1" + "01" * (int(n / 2) - 1) + "1", length=n
)
z = pattern_fn(b)

print(
    f"""
        {sum([x[i]*2**((len(x)-1)-i) for i in range(len(x))])}
        +
        {sum([y[i]*2**((len(y)-1)-i) for i in range(len(y))])}
        =
        {sum([z[i]*2**((len(z)-1)-i) for i in range(len(z))])}
    """
)

plot_circuit(addition)
print("hi")
