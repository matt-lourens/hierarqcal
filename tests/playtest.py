# sandbox file for adhoc tests
# %%
from os import write
from quantum_estimators.qcnn import Qcnn_Classifier

# %% Quantum - Pennylane

# %% Data
import numpy as np
from collections import namedtuple
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def filter_to_binary(samples, target_pair):
    train_filter = np.where(
        (samples.y_train == target_pair[0]) | (samples.y_train == target_pair[1])
    )
    test_filter = np.where(
        (samples.y_test == target_pair[0]) | (samples.y_test == target_pair[1])
    )
    X_train_filtered, X_test_filtered = (
        samples.X_train[train_filter],
        samples.X_test[test_filter],
    )
    y_train_filtered, y_test_filtered = (
        samples.y_train[train_filter],
        samples.y_test[test_filter],
    )

    y_train_filtered = np.where(y_train_filtered == target_pair[1], 1, 0)
    y_test_filtered = np.where(y_test_filtered == target_pair[1], 1, 0)

    return Samples(X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered)


Samples = namedtuple("Samples", ["X_train", "y_train", "X_test", "y_test"])

random_state = 1
test_size = 0.3
target_pair = (0, 1)  # setosa,versicolor

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state,
)
# store samples in named tuple
samples = Samples(X_train, y_train, X_test, y_test)
# focus only on specified target pair
samples_filtered = filter_to_binary(samples, target_pair)

# %% preprocessing
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

scaler_params = {"feature_range": [0, np.pi / 2]}
scaler = (
    "minmax",
    preprocessing.MinMaxScaler(**scaler_params),
)
# Add all preprocessing steps to this list
steps_list = [scaler]
pipeline = Pipeline(steps_list)


# %% Model
quantum_fn_mapping = {
    "module_import_name": "quantum_pennylane",  # my_code.some_feature.quantum
    "quantum_execution:": "quantum_node",
    "layer_fn_mapping": {1: (("U", 10), ("V", 2))},
}
n_wires = X.shape
model = Qcnn_Classifier(
    quantum_mapping=quantum_fn_mapping,
    n_iter=50,
    batch_size=50,
    learning_rate=0.01,
    optimizer="adam",
    n_wires=n_wires,
    s_c=1,
    pool_filter="right",
    s_p=0,
    seed=1,
)

pipeline.steps.add("qcnn", model())

# %% Execute
from sklearn.metrics import confusion_matrix

pipeline.fit(samples_filtered.X_train, samples_filtered.y_train)
y_hat = pipeline.predict(samples_filtered.X_test)
cf_matrix = confusion_matrix(samples_filtered.y_test, y_hat)
# %%
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OnveVsOneClassifier

sclaer = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
clf = LogisticRegression(penalty="l1")
clf.fit(X, y)
y_pred = clf.predict(X_test)

ovo_lr = OnveVsOneClassifier(LogisticRegression(penalty="l1"))

# %%
import pennylane as qml

dev = qml.device("default.qubit", wires=1)


@qml.qnode(dev)
def circuit(x):
    qml.RX(x, wires=0)
    return qml.expval(qml.PauliZ(0))


# %%
def circuit(x):
    qml.RX(x, wires=0)
    return qml.expval(qml.PauliZ(0))


dev = qml.device("default.qubit", wires=1)
qnode = qml.QNode(circuit, dev)
# %% =================================================================================
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
x_test_circ = tfq.convert_to_tensor([qubit_encoding(x) for x in samples_tfd.X_test])

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
model.compile(optimizer='Adam', loss='mse')
model.fit(x=samples_circ.X_train,
          y=samples_circ.y_train,
          epochs=10)
# %%
model.summary()
print(model.trainable_variables)



# class QcnnArchitecture:
#     def __init__(self, graphs=[], n_q=8, architecture_strategy=None, **kwargs) -> None:
#         self.n_q = n_q
#         # sequence of graphs
#         if self.architecture_strategy == "binary_tree_r":
#             self.graphs = self._get_binary_tree_r_graphs(**kwargs)
#         else:
#             self.graphs = graphs

#     def append(self, value: tuple) -> None:
#         self.graphs.append(value)
#         self.n_layers += 1
#         return self.graphs

#     def _get_binary_tree_r_graphs(self):
#         """ """
#         if type(self.pool_filter) is str:
#             # Mapping words to the filter type
#             if self.pool_filter == "left":
#                 # 0 1 2 3 4 5 6 7
#                 # x x x x
#                 self.pool_filter = lambda arr: arr[0 : len(arr) // 2 : 1]
#             elif self.pool_filter == "right":
#                 # 0 1 2 3 4 5 6 7
#                 #         x x x x
#                 self.pool_filter = lambda arr: arr[len(arr) : len(arr) // 2 - 1 : -1]
#             elif self.pool_filter == "even":
#                 # 0 1 2 3 4 5 6 7
#                 # x   x   x   x
#                 self.pool_filter = lambda arr: arr[0::2]
#             elif self.pool_filter == "odd":
#                 # 0 1 2 3 4 5 6 7
#                 #   x   x   x   x
#                 self.pool_filter = lambda arr: arr[1::2]
#             elif self.pool_filter == "inside":
#                 # 0 1 2 3 4 5 6 7
#                 #     x x x x
#                 self.pool_filter = (
#                     lambda arr: arr[
#                         len(arr) // 2
#                         - len(arr) // 4 : len(arr) // 2
#                         + len(arr) // 4 : 1
#                     ]
#                     if len(arr) > 2
#                     else [arr[1]]
#                 )  # inside
#             elif self.pool_filter == "outside":
#                 # 0 1 2 3 4 5 6 7
#                 # x x         x x
#                 self.pool_filter = (
#                     lambda arr: [
#                         item
#                         for item in arr
#                         if not (
#                             item
#                             in arr[
#                                 len(arr) // 2
#                                 - len(arr) // 4 : len(arr) // 2
#                                 + len(arr) // 4 : 1
#                             ]
#                         )
#                     ]
#                     if len(arr) > 2
#                     else [arr[0]]
#                 )  # outside

#         graphs = {}
#         layer = 1
#         Qc_l = [i + 1 for i in range(self.n_q)]  # We label the nodes from 1 to n
#         Qp_l = Qc_l.copy()
#         while len(Qc_l) > 1:

#             nq_avaiable = len(Qc_l)
#             mod_nq = lambda x: x % nq_avaiable
#             Ec_l = [(Qc_l[i], Qc_l[mod_nq(i + self.s_c)]) for i in range(nq_avaiable)]
#             if len(Ec_l) == 2 and Ec_l[0][0:] == Ec_l[1][1::-1]:
#                 Ec_l = [Ec_l[0]]
#             measured_q = self.pool_filter(Qc_l)
#             remaining_q = [q for q in Qc_l if not (q in measured_q)]
#             Ep_l = [
#                 (measured_q[i], remaining_q[(i + self.s_p) % len(remaining_q)])
#                 for i in range(len(measured_q))
#             ]
#             # Convolution graph
#             C_l = (Qc_l, Ec_l)
#             # Pooling graph
#             P_l = (Qp_l, Ep_l)
#             # Graph for layer
#             G_l = (C_l, P_l)
#             graphs[layer] = G_l
#             # set avaiable qubits for next layer
#             layer = layer + 1
#             Qc_l = [j for (i, j) in Ep_l]
#             Qp_l = Qc_l.copy()
#         return graphs


# class QConv(keras.layers.Layer):
#     def __init__(
#         self,
#         layer=1,
#         n_q=8,
#         stride=1,
#         unitary=None,
#         n_theta=None,
#         theta_init_seed=None,
#         theta_range=(0, 2 * np.pi),
#         graph=None,
#         e_pl=None,
#         total_coef_before=0,
#         name="QConv",
#         **kwargs,
#     ):
#         super(QConv, self).__init__(name=name, **kwargs)
#         self.stride = stride
#         self.unitary = U if unitary is None else unitary
#         self.n_theta = 10 if unitary is None else n_theta
#         # TODO handle case of unitary provided but not n_theta
#         self.theta_init_seed = theta_init_seed
#         self.theta_range = theta_range
#         self.graph = graph

#         self.layer = layer
#         if self.graph is None:
#             self.graph = self._build_conv_graph(e_pl)
#         else:
#             self.graph = graph
#         self.circuit, self.symbols, self.total_coef_after = self._construct_circuit(
#             total_coef_before
#         )

#     def _build_conv_graph(self, e_pl):
#         # use stride and n_q to determine convolution
#         if self.layer == 1:
#             Qc_l = [i + 1 for i in range(self.n_q)]
#         else:
#             Qc_l = [j for (i, j) in e_pl]
#         nq_avaiable = len(Qc_l)
#         mod_nq = lambda x: x % nq_avaiable
#         Ec_l = [(Qc_l[i], Qc_l[mod_nq(i + self.s_c)]) for i in range(nq_avaiable)]
#         return (Qc_l, Ec_l)

#     def _construct_circuit(self, total_coef_before):
#         circuit = cirq.Circuit()
#         symbols = ()
#         # graph is a tuple of nodes then edges: (Qc_l,Ec_l)
#         E_cl = self.graph[1]
#         if self.n_theta > 0:
#             layer_symbols_c = sympy.symbols(
#                 f"x_{total_coef_before}:{total_coef_before + self.n_theta}"
#             )
#             symbols += layer_symbols_c
#             total_coef_after = total_coef_before + self.n_theta
#         # Convolution Operation
#         for bits in E_cl:
#             if self.n_theta > 0:
#                 circuit.append(self.unitary(bits, layer_symbols_c))
#             else:
#                 # If the circuit has no paramaters then the only argument is bits
#                 circuit.append(self.unitary(bits))
#         return circuit, symbols, total_coef_after