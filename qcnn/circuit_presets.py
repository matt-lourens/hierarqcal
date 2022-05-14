# This module contains the set of unitary ansatze that will be used to benchmark the performances of Quantum Convolutional Neural Network (QCNN) in QCNN.ipynb module
import pennylane as qml
from math import log2
import numpy as np
from sympy import true
import itertools as it

# Specify options


# Name of circuit function from unitary.py along with param count
CIRCUIT_OPTIONS = {
    "U_TTN": 2,
    "U_5": 10,
    "U_6": 10,
    "U_9": 2,
    "U_13": 6,
    "U_14": 6,
    "U_15": 4,
    "U_SO4": 6,
    "U_SU4": 15,
}

POOLING_OPTIONS = {"psatz1": 2, "psatz2": 0, "psatz3": 3}

# TODO randomized pooling
def get_wire_combos(n_wires, c_step, pool_pattern, p_step=0, wire_to_cut=1):
    """n_wires = 8
        c_step = 1
        p_step = 0
        pool_pattern = "eo_even"
        wire_to_cut = 0
        ^^produces the original papaers pattern

    Args:
        n_wires ([type]): [description]
        c_step ([type]): [description]
        pool_pattern ([type]): [description]
        p_step (int, optional): [description]. Defaults to 0.
        wire_to_cut (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """
    if pool_pattern == "left":
        # 0 1 2 3 4 5 6 7
        # x x x x
        pool_filter = lambda arr: arr[0 : len(arr) // 2 : 1]
    elif pool_pattern == "right":
        # 0 1 2 3 4 5 6 7
        #         x x x x
        pool_filter = lambda arr: arr[len(arr) : len(arr) // 2 - 1 : -1]
    elif pool_pattern == "eo_even":
        # 0 1 2 3 4 5 6 7
        # x   x   x   x
        pool_filter = lambda arr: arr[0::2]
    elif pool_pattern == "eo_odd":
        # 0 1 2 3 4 5 6 7
        #   x   x   x   x
        pool_filter = lambda arr: arr[1::2]
    elif pool_pattern == "inside":
        # 0 1 2 3 4 5 6 7
        #     x x x x
        pool_filter = lambda arr: arr[
            len(arr) // 2 - len(arr) // 4 : len(arr) // 2 + len(arr) // 4 : 1
        ]  # inside
    elif pool_pattern == "outside":
        # 0 1 2 3 4 5 6 7
        # x x         x x
        pool_filter = lambda arr: [
            item
            for item in arr
            if not (
                item
                in arr[
                    len(arr) // 2 - len(arr) // 4 : len(arr) // 2 + len(arr) // 4 : 1
                ]
            )
        ]  # outside
    wire_combos = {}
    wires = range(n_wires)
    for layer_ind, i in zip(
        range(int(log2(n_wires))), range(int(log2(n_wires)), 0, -1)
    ):
        conv_size = 2 ** i
        circle_n = lambda x: x % conv_size
        wire_combos[f"c_{layer_ind+1}"] = [
            (wires[x], wires[circle_n(x + c_step)]) for x in range(conv_size)
        ]
        if (i == 1) and (len(wire_combos[f"c_{layer_ind+1}"]) > 1):
            wire_combos[f"c_{layer_ind+1}"] = [wire_combos[f"c_{layer_ind+1}"][0]]

        tmp_pool_selection = pool_filter(wire_combos[f"c_{layer_ind+1}"])
        cut_wires = [x[wire_to_cut] for x in tmp_pool_selection]
        wires = [wire for wire in wires if not (wire in cut_wires)]
        p_circle_n = lambda x: x % len(cut_wires)
        wire_combos[f"p_{layer_ind+1}"] = [
            (cut_wires[p_circle_n(x + p_step)], wires[x]) for x in range(len(cut_wires))
        ]
        # wire_combos[f"p_{layer_ind+1}"] = pool_filter(wire_combos[f"c_{layer_ind+1}"])
        if len(wire_combos[f"p_{layer_ind+1}"]) == 0:
            wire_combos[f"p_{layer_ind+1}"] = [wire_combos[f"c_{layer_ind+1}"][0]]
    return wire_combos


def get_qcnn_graphs(n_wires, c_step, pool_pattern, p_step=0):
    """ """
    if type(pool_pattern) is str:
        # Mapping words to the filter type
        if pool_pattern == "left":
            # 0 1 2 3 4 5 6 7
            # x x x x
            pool_filter = lambda arr: arr[0 : len(arr) // 2 : 1]
        elif pool_pattern == "right":
            # 0 1 2 3 4 5 6 7
            #         x x x x
            pool_filter = lambda arr: arr[len(arr) : len(arr) // 2 - 1 : -1]
        elif pool_pattern == "eo_even":
            # 0 1 2 3 4 5 6 7
            # x   x   x   x
            pool_filter = lambda arr: arr[0::2]
        elif pool_pattern == "eo_odd":
            # 0 1 2 3 4 5 6 7
            #   x   x   x   x
            pool_filter = lambda arr: arr[1::2]
        elif pool_pattern == "inside":
            # 0 1 2 3 4 5 6 7
            #     x x x x
            pool_filter = lambda arr: arr[
                len(arr) // 2 - len(arr) // 4 : len(arr) // 2 + len(arr) // 4 : 1
            ]  # inside
        elif pool_pattern == "outside":
            # 0 1 2 3 4 5 6 7
            # x x         x x
            pool_filter = lambda arr: [
                item
                for item in arr
                if not (
                    item
                    in arr[
                        len(arr) // 2
                        - len(arr) // 4 : len(arr) // 2
                        + len(arr) // 4 : 1
                    ]
                )
            ]  # outside
    else:
        pool_filter = pool_pattern

    graphs = {}
    layer = 1
    Qc_l = [i + 1 for i in range(n_wires)]
    Qp_l = Qc_l.copy()
    while len(Qc_l) > 1:

        nq_avaiable = len(Qc_l)
        mod_nq = lambda x: x % nq_avaiable
        Ec_l = [(Qc_l[i], Qc_l[mod_nq(i + c_step)]) for i in range(nq_avaiable)]
        if len(Ec_l) == 2 and Ec_l[0][0:] == Ec_l[1][1::-1]:
            # TODO improve this, the issue is (1,2) and (2,1) with this logic, there might be a better
            # TODO way to traverse the graph in a general way
            Ec_l = [Ec_l[0]]
        measured_q = pool_filter(Qc_l)
        remaining_q = [q for q in Qc_l if not (q in measured_q)]
        Ep_l = [
            (measured_q[i], remaining_q[(i + p_step) % len(remaining_q)])
            for i in range(len(measured_q))
        ]
        # Convolution graph
        C_l = (Qc_l, Ec_l)
        # Pooling graph
        P_l = (Qp_l, Ep_l)
        # Graph for layer
        G_l = (C_l, P_l)
        graphs[layer] = G_l
        # set avaiable qubits for next layer
        layer = layer + 1
        Qc_l = [j for (i, j) in Ep_l]
        Qp_l = Qc_l.copy()
    return graphs


def c_1(circuit, params):
    circuit(params, wires=[0, 7])
    for i in range(0, 8, 2):
        circuit(params, wires=[i, i + 1])
    for i in range(1, 7, 2):
        circuit(params, wires=[i, i + 1])


def c_2(circuit, params):
    circuit(params, wires=[0, 6])
    circuit(params, wires=[0, 2])
    circuit(params, wires=[4, 6])
    circuit(params, wires=[2, 4])


def c_3(circuit, params):
    circuit(params, wires=[0, 4])


# Pooling layers
def p_1(circuit, params):
    for i in range(0, 8, 2):
        circuit(params, wires=[i + 1, i])


def p_2(circuit, params):
    circuit(params, wires=[2, 0])
    circuit(params, wires=[6, 4])


def p_3(circuit, params):
    circuit(params, wires=[0, 4])


# Unitary Ansatze for Convolutional Layer
def U_TTN(params, wires):
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])


def U_5(params, wires):  # 10 params
    qml.RX(params[0], wires=wires[0])
    qml.RX(params[1], wires=wires[1])
    qml.RZ(params[2], wires=wires[0])
    qml.RZ(params[3], wires=wires[1])
    qml.CRZ(params[4], wires=[wires[1], wires[0]])
    qml.CRZ(params[5], wires=[wires[0], wires[1]])
    qml.RX(params[6], wires=wires[0])
    qml.RX(params[7], wires=wires[1])
    qml.RZ(params[8], wires=wires[0])
    qml.RZ(params[9], wires=wires[1])


def U_5(params, wires):  # 8 parmas
    qml.RX(params[0], wires=wires[0])
    qml.RX(params[1], wires=wires[1])
    qml.RZ(params[2], wires=wires[0])
    qml.RZ(params[3], wires=wires[1])
    qml.CRZ(params[4], wires=[wires[1], wires[0]])
    qml.CRZ(params[5], wires=[wires[0], wires[1]])
    qml.RX(params[6], wires=wires[0])
    qml.RX(params[7], wires=wires[1])


def U_6(params, wires):  # 10 params
    qml.RX(params[0], wires=wires[0])
    qml.RX(params[1], wires=wires[1])
    qml.RZ(params[2], wires=wires[0])
    qml.RZ(params[3], wires=wires[1])
    qml.CRX(params[4], wires=[wires[1], wires[0]])
    qml.CRX(params[5], wires=[wires[0], wires[1]])
    qml.RX(params[6], wires=wires[0])
    qml.RX(params[7], wires=wires[1])
    qml.RZ(params[8], wires=wires[0])
    qml.RZ(params[9], wires=wires[1])


def U_9(params, wires):  # 2 params
    qml.Hadamard(wires=wires[0])
    qml.Hadamard(wires=wires[1])
    qml.CZ(wires=[wires[0], wires[1]])
    qml.RX(params[0], wires=wires[0])
    qml.RX(params[1], wires=wires[1])


def U_13(params, wires):  # 6 params
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CRZ(params[2], wires=[wires[1], wires[0]])
    qml.RY(params[3], wires=wires[0])
    qml.RY(params[4], wires=wires[1])
    qml.CRZ(params[5], wires=[wires[0], wires[1]])


def U_14(params, wires):  # 6 params
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CRX(params[2], wires=[wires[1], wires[0]])
    qml.RY(params[3], wires=wires[0])
    qml.RY(params[4], wires=wires[1])
    qml.CRX(params[5], wires=[wires[0], wires[1]])


def U_15(params, wires):  # 4 params
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(params[2], wires=wires[0])
    qml.RY(params[3], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])


def U_SO4(params, wires):  # 6 params
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[2], wires=wires[0])
    qml.RY(params[3], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[4], wires=wires[0])
    qml.RY(params[5], wires=wires[1])


def U_SU4(params, wires):  # 15 params arbitrary 2 Qbit unitary
    qml.U3(params[0], params[1], params[2], wires=wires[0])
    qml.U3(params[3], params[4], params[5], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[6], wires=wires[0])
    qml.RZ(params[7], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(params[8], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.U3(params[9], params[10], params[11], wires=wires[0])
    qml.U3(params[12], params[13], params[14], wires=wires[1])


# Pooling Layer
def psatz1(
    params, wires
):  # 2 params # TODO classical post processing, quantum teleportation protocall Nielsung and Chuang tunneling protocall 1.3.7 p 26
    qml.CRZ(params[0], wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])
    qml.CRX(params[1], wires=[wires[0], wires[1]])


def psatz2(wires):  # 0 params
    qml.CRZ(wires=[wires[0], wires[1]])


def psatz3(*params, wires):  # 3 params
    qml.CRot(*params, wires=[wires[0], wires[1]])


# %%


# %%
