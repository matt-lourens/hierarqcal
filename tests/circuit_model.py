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






# %%
import quantum_cirq_3
import importlib

importlib.reload(quantum_cirq_3)
from quantum_cirq_3 import Qcnn_Classifier

def qubit_encoding(x):
    circuit = cirq.Circuit()
    for i, value in enumerate(x):
        qubit = cirq.LineQubit(i)
        circuit.append(cirq.rx(x[i]).on(qubit))

    return circuit


def U(bits, symbols=None):
    circuit = cirq.Circuit()
    q0, q1 = cirq.LineQubit(bits[0]), cirq.LineQubit(bits[1])
    circuit += cirq.rz(symbols[0]).on(q1).controlled_by(q0)
    circuit += cirq.rz(symbols[1]).on(q0).controlled_by(q1)
    return circuit


def V(bits, symbols=None):
    circuit = cirq.Circuit()
    q0, q1 = cirq.LineQubit(bits[0]), cirq.LineQubit(bits[1])
    circuit += cirq.CNOT(q0, q1)
    return circuit

x_train_circ = tfq.convert_to_tensor([qubit_encoding(x) for x in samples_tfd.X_train])
x_test_circ =  tfq.convert_to_tensor([qubit_encoding(x) for x in samples_tfd.X_test])

samples_circ = Samples(
    x_train_circ, samples_tfd.y_train, x_test_circ, samples_tfd.y_test
)

original_inputs = tf.keras.Input(
    shape=tf.shape(samples_tfd.X_train), dtype=tf.dtypes.float64
)

# This is needed because of Note here:
# https://www.tensorflow.org/quantum/api_docs/python/tfq/layers/Expectation
input_circuits = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
# encoding_layer = tfq.layers.AddCircuit()(input_circuits, prepend=samples_circ.X_train)
qcnn_circuit = Qcnn_Classifier(
    convolution_mapping={1: (U, 2)}, pooling_mapping={1: (V, 0)}
)(input_circuits)
model = tf.keras.Model(inputs=[input_circuits], outputs=[qcnn_circuit])
# quantum_execution = tfq.layers.PQC()()
# circuit = qcnn_1(samples_tfd.X_train)
# circuit


# %%
model.compile(optimizer='Adam', loss='binary_crossentropy')
model.fit(x=samples_circ.X_train,
          y=samples_circ.y_train,
          epochs=10)
# %%
model.summary()
print(model.trainable_variables)