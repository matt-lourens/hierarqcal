# sandbox file for adhoc tests
# %%
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
    "quantum_execution:":"quantum_node",
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
# %%
