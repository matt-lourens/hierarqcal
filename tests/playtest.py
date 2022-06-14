# sandbox file for adhoc tests
# %%
from quantum_estimators.qcnn import Qcnn_Classifier

# %% Quantum - Pennylane
import pennylane as qml
from pennylane.templates.embeddings import AngleEmbedding


class Quantum_Pennylane:
    def Quantum_Pennylane(
        self,
        n_wires=8,
        device="default.qubit",
        interface="torch",
        layer_fn_mapping={1: ("U", "V")},
    ):
        self.n_wires = n_wires
        self.device = qml.device(device, wires=self.n_wires)
        self.interface = interface
        self.layer_fn_mapping = layer_fn_mapping

    def execute(self):
        return qml.QNode(self.subroutine, self.device)

    def subroutine(X, classifier):
        if getattr(classifier, "numpy", False):
            # If classifier needs to be deserialized
            classifier = classifier.numpy()
        # Encode data
        n_wires = X.shape[0]
        AngleEmbedding(X, wires=range(n_wires), rotation="Y")
        # Evaluate circuit
        classifier._evaluate()
        # Obtain probability
        result = qml.probs(wires=classifier.response_wire_)

        return result

    def U(params, wires):
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

    def V(params, wires):
        qml.CRZ(params[0], wires=[wires[0], wires[1]])
        qml.PauliX(wires=wires[0])
        qml.CRX(params[1], wires=[wires[0], wires[1]])


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
n_wires = X.shape
model = Qcnn_Classifier(
    Quantum_class=Quantum_Pennylane,
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
