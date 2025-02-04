import pennylane as qml
from hierarqcal import Qinit, Qcycle, Qunitary, Qmask
from hierarqml.utils import hierq_gates
import numpy as np


def ansatz(bits, symbols):  # 10 params
    qml.RX(symbols[0], wires=bits[0])
    qml.RX(symbols[1], wires=bits[1])
    qml.RZ(symbols[2], wires=bits[0])
    qml.RZ(symbols[3], wires=bits[1])
    qml.CRZ(symbols[4], wires=[bits[1], bits[0]])
    qml.CRZ(symbols[5], wires=[bits[0], bits[1]])
    qml.RX(symbols[6], wires=bits[0])
    qml.RX(symbols[7], wires=bits[1])
    qml.RZ(symbols[8], wires=bits[0])
    qml.RZ(symbols[9], wires=bits[1])


def get_motif(n):
    qcnn = Qinit(n) + (
        Qcycle(
            stride=1,
            step=1,
            offset=0,
            mapping=Qunitary(ansatz, n_symbols=10, arity=2),
            share_weights=True,
        )
        + Qmask("!*", mapping=hierq_gates["CNOT"])
    ) * int(np.log2(n))
    return qcnn
