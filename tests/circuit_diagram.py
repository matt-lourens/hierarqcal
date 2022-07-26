# %%
import tensorflow as tf
import tensorflow_quantum as tfq
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
import cirq
import sympy
import numpy as np
import pandas as pd
from collections import namedtuple

# visualization tools
# %matplotlib inline
import matplotlib.pyplot as plt


Samples = namedtuple("Samples", ["X_train", "y_train", "X_test", "y_test"])

path = "/home/matt/dev/projects/quantum_estimators/tests/features_30_sec.csv"
raw = pd.read_csv(path)
target = "label"
columns_to_remove = ["filename", "length", target]
y = raw.loc[:, target]
X = raw.drop(columns_to_remove, axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
)
samples = Samples(X_train, y_train, X_test, y_test)

target_pair = ["rock", "reggae"]
train_filter = np.where(
    (samples.y_train == target_pair[0]) | (samples.y_train == target_pair[1])
)
test_filter = np.where(
    (samples.y_test == target_pair[0]) | (samples.y_test == target_pair[1])
)
X_train_filtered, X_test_filtered = (
    samples.X_train.iloc[train_filter],
    samples.X_test.iloc[test_filter],
)
y_train_filtered, y_test_filtered = (
    samples.y_train.iloc[train_filter],
    samples.y_test.iloc[test_filter],
)

y_train_filtered = np.where(y_train_filtered == target_pair[1], 1, 0)
y_test_filtered = np.where(y_test_filtered == target_pair[1], 1, 0)

samples_filtered = Samples(
    X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered
)


# setup pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

scaler = scaler = (
    "minmax",
    preprocessing.MinMaxScaler(feature_range=[0, np.pi / 2]),
)
selection = selection = (
    "tree",
    SelectFromModel(
        ExtraTreesClassifier(n_estimators=50),
        max_features=8,
    ),
)

pipeline = Pipeline([scaler, selection])
pipeline.fit(samples_filtered.X_train, samples_filtered.y_train)

# Transform data
X_train_tfd = pipeline.transform(samples_filtered.X_train)
X_test_tfd = pipeline.transform(samples_filtered.X_test)
samples_tfd = Samples(
    X_train_tfd, samples_filtered.y_train, X_test_tfd, samples_filtered.y_test
)


def qubit_encoding(x):
    circuit = cirq.Circuit()
    for i, value in enumerate(x):
        qubit = cirq.LineQubit(i)
        circuit.append(cirq.rx(x[i]).on(qubit))

    return circuit


x_train_circ = [qubit_encoding(x) for x in samples_tfd.X_train]
x_test_circ = [qubit_encoding(x) for x in samples_tfd.X_test]

samples_circ = Samples(
    x_train_circ, samples_tfd.y_train, x_test_circ, samples_tfd.y_test
)
# %%
import quantum_cirq_3
import importlib

importlib.reload(quantum_cirq_3)
from quantum_cirq_3 import Qcnn_Classifier


def U(bits, symbols=None):
    circuit = cirq.Circuit()
    q0, q1 = cirq.LineQubit(bits[0]), cirq.LineQubit(bits[1])
    circuit += cirq.rz(symbols[0]).on(q1).controlled_by(q0)
    # circuit += cirq.rz(symbols[1]).on(q0).controlled_by(q1)
    return circuit


def V(bits, symbols=None):
    circuit = cirq.Circuit()
    q0, q1 = cirq.LineQubit(bits[0]), cirq.LineQubit(bits[1])
    circuit += cirq.CNOT(q0, q1)
    return circuit


qcnn_1 = Qcnn_Classifier(convolution_mapping={1: (U, 1)}, pooling_mapping={1: (V, 0)})
circuit = qcnn_1(samples_tfd.X_train)
circuit

# %% SVG
from cirq.contrib.svg import SVGCircuit

SVGCircuit(circuit)
# %%
import cirq.contrib.qcircuit as ccq
import sympy

pooling_circuit = V((1, 2))
symbols = sympy.symbols(f"\\theta_{{{0}:{0 + 1}}}")
convolution_circuit = U((1, 2), symbols)
convolution_circuit_tex = ccq.circuit_to_latex_using_qcircuit(convolution_circuit)
pooling_circuit_tex = ccq.circuit_to_latex_using_qcircuit(pooling_circuit)
# a = ccq.escape_text_for_latex(a)
# ccq.circuit_to_pdf_using_qcircuit_via_tex(circuit, "circuit.pdf")
with open("/home/matt/dev/projects/quantum_estimators/diagrams/convolution_circuit.txt", "w") as f:
    f.write(convolution_circuit_tex)

with open("/home/matt/dev/projects/quantum_estimators/diagrams/pooling_circuit.txt", "w") as f:
    f.write(pooling_circuit_tex)

# %%
import itertools as it
from functools import reduce
import cirq.contrib.qcircuit as ccq
from cirq.contrib.svg import SVGCircuit

for s_c, s_p, pool_filter in it.product(
    [1, 3, 5, 7],
    [0, 1, 2, 3],
    ["right", "left", "even", "odd", "inside", "outside"],
):
    qcnn_1 = Qcnn_Classifier(
        n_q=8,
        s_c=s_c,
        s_p=s_p,
        pool_filter=pool_filter,
        convolution_mapping={1: (U, 1)},
        pooling_mapping={1: (V, 0)},
    )
    circuit = qcnn_1(samples_tfd.X_train)
    circuit.moments[0] = reduce(lambda x, y: x + y, circuit.moments[0:8:2])
    circuit.moments[1] = reduce(lambda x, y: x + y, circuit.moments[1:8:2])
    circuit.moments[12] = circuit.moments[12] + circuit.moments[14]
    indexes = [2, 3, 4, 5, 6, 7, 14]
    for idx in sorted(indexes, reverse=True):
        del circuit.moments[idx]
    SVGCircuit(circuit)
    a = ccq.circuit_to_latex_using_qcircuit(circuit)
    # a = ccq.escape_text_for_latex(a)
    # ccq.circuit_to_pdf_using_qcircuit_via_tex(circuit, "circuit.pdf")
    with open(
        f"/home/matt/dev/projects/quantum_estimators/diagrams/all_aligned.txt",
        "a",
    ) as f:
        f.write(f"\\text{{{s_c}-{s_p}-{pool_filter}}}\\newline\n" f"{a}\\newline\n")
# a = it.product([1,3,5,7],[0,1,2,3],["right", "left","eo_even","eo_odd","inside","outside"])
# %%
