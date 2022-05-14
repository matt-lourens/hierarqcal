import time
import multiprocessing
from functools import partial
import numpy as np
import autograd.numpy as anp
from joblib import delayed, Parallel, wrap_non_picklable_objects
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.utils.multiclass import type_of_target
import pennylane as qml

# import qiskit
# from qiskit import IBMQ, Aer
# from qiskit.providers.aer.noise import NoiseModel
# from qiskit.providers.aer.noise import depolarizing_error
from data_handler import save_json

# from pennylane_cirq import ops
import torch
from torch import nn

# from qiskit.providers.aer.noise.device import basic_device_noise_model
from embedding import apply_encoding
import circuit_presets
from circuit_presets import CIRCUIT_OPTIONS, POOLING_OPTIONS, get_qcnn_graphs


class Qcnn_Classifier(BaseEstimator, ClassifierMixin):
    """
    from qcnn_estimator import Qcnn_Classifier
    from sklearn.utils.estimator_checks import check_estimator
    estimator = Qcnn_Classifier()
    check_estimator(estimator)
    """

    def __init__(
        self,
        n_iter=25,
        learning_rate=0.01,
        batch_size=25,
        optimizer="adam",
        cost="cross_entropy",
        encoding_type="Angle",
        encoding_kwargs={},
        layer_defintion=("U_5", "psatz1", [8, 1, "eo_even"]),
        noise=None,
        seed=1,
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.cost = cost
        self.encoding_type = encoding_type
        self.layer_defintion = layer_defintion
        self.encoding_kwargs = encoding_kwargs
        self.noise = noise
        self.seed = seed
        # find place for these parameters set
        # params = qml_np.random.randn(qcnn_structure.paramater_count)

    def _more_tags(self):
        return {
            "binary_only": True,
        }

    def fit(self, X, y):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, ensure_2d=False)

        # Construct Quantum Convolutional structure
        self.layer_dict_ = self._construct_layer_dict(self.layer_defintion)
        # Sort the layers according to the order provided
        self._sort_layer_dict_by_order()
        # Get coefficient information
        self.coef_count_, self.coef_indices_ = self._get_coef_information()
        # Initialize Coefficients
        if self.seed:
            torch.manual_seed(self.seed)
        coefficients = torch.rand(self.coef_count_, requires_grad=True)
        tmp_layer_info = [
            (layer_name, layer.layer_order)
            for layer_name, layer in self.layer_dict_.items()
        ]
        # Gets the layer name with the max order
        final_layer = max(tmp_layer_info, key=lambda item: item[1])[0]
        if self.layer_dict_[final_layer].wire_pattern == None:
            # The default is 4, this mostly makes it backwards compatible
            self.response_wire_ = 4
        else:
            self.response_wire_ = self.layer_dict_[final_layer].wire_pattern[0][1]
        # Set paramaters that saves training information
        self.train_history_ = {"Iteration": [], "Cost": [], "Time": []}
        self.test_history_ = {"Iteration": [], "Cost": []}
        self.coef_history_ = {}

        # Set optimizer
        if self.optimizer == "nesterov":
            opt = qml.NesterovMomentumOptimizer(stepsize=self.learning_rate)
        elif self.optimizer == "adam":
            opt = torch.optim.Adam([coefficients], lr=self.learning_rate)
        else:
            raise NotImplementedError(
                f"There is no implementation for optimizer: {self.optimizer}"
            )

        for it in range(self.n_iter):
            # Sample a batch from the training set
            # TODO oversampling manually implemented but should be configureable
            # batch_train_index = np.random.randint(X.shape[0], size=self.batch_size)
            # X_train_batch = X[batch_train_index]
            # y_train_batch = torch.from_numpy(np.array(y)[batch_train_index])
            # randomly sample y==1 values
            batch_y1 = np.where(y == 1)[0][
                np.random.randint(
                    np.where(y == 1)[0].shape[0], size=self.batch_size // 2
                )
            ]
            # randomly sample y==0 values
            batch_y0 = np.where(y == 0)[0][
                np.random.randint(
                    np.where(y == 0)[0].shape[0],
                    size=self.batch_size - self.batch_size // 2,
                )
            ]
            batch_train_index = np.append(batch_y1, batch_y0)
            X_train_batch = X[batch_train_index]
            y_train_batch = torch.from_numpy(np.array(y)[batch_train_index])

            # Run model and get cost
            t0 = time.time()
            opt.zero_grad()
            loss = self.coefficient_based_loss(
                coefficients, X_train_batch, y_train_batch
            )
            loss.backward()
            opt.step()
            # t1 = time.time()
            # print(t1 - t0)
            # coefficients, cost_train = opt.step_and_cost(
            #     lambda current_coef: self.coefficient_based_loss(
            #         current_coef, X_train_batch, y_train_batch
            #     ),
            #     coefficients,
            # )
            t1 = time.time()
            # print(t1 - t0)
            self.train_history_["Iteration"].append(it)
            self.train_history_["Cost"].append(loss.detach().numpy().tolist())
            self.train_history_["Time"].append(t1 - t0)
            self.coef_history_[it] = coefficients.detach().numpy().tolist()
            # save_json(
            #     "/home/matt/dev/projects/quantum-cnn/reports/execution_time/train_history.json",
            #     self.train_history_,
            # )
            # save_json(
            #     "/home/matt/dev/projects/quantum-cnn/reports/execution_time/coef_history.json",
            #     self.coef_history_,
            # )

        best_iteration = self.train_history_["Iteration"][
            np.argmin(self.train_history_["Cost"])
        ]
        best_coefficients = self.coef_history_[best_iteration]
        # Set model coefficient corresponding to iteration that had lowest loss
        self.coef_ = np.array(best_coefficients)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def coefficient_based_loss(self, coef, X, y):
        """Used for training"""
        self.coef_ = coef
        loss = self.score(X, y, return_loss=True, require_tensor=True)
        return loss

    def predict(self, X, cutoff=0.5):
        """A reference implementation of a predicting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")
        y_hat = self.predict_proba(X)
        if self.cost == "cross_entropy":
            """If the cost function is cross entropy then predictions are returned
            In the shpae:[(p_0, p_1)]"""
            y_hat_clf = [np.where(x == max(x))[0][0] for x in y_hat]
        else:
            """
            Choose based on cutoff
            """
            y_hat_clf = np.array([1 if y_p >= cutoff else 0 for y_p in y_hat])

        return y_hat_clf

    def predict_proba(self, X, require_tensor=False):
        if require_tensor:
            y_hat = [quantum_node(x, self) for x in X]
            y_hat = torch.stack(y_hat)
            # parallel
            # pool = multiprocessing.Pool()
            # # y_hat = pool.map(partial(qcnn_estimator.quantum_node,classifier=self), X)
            # y_hat = Parallel(n_jobs=2)(
            #     delayed(partial(quantum_node, classifier=self))(x) for x in X
            # )
        else:
            y_hat = np.array([quantum_node(x, self).numpy() for x in X])
            # pool = multiprocessing.Pool()
            # y_hat = pool.map(quantum_node, [{"x": x, "classifier": self} for x in X])
        return y_hat

    def score(self, X, y, return_loss=False, **kwargs):
        """Returns the score (bigger is better) of the current model. Can specify to return
        the loss (which is the negative of the score). Reason to return loss is for things like training
        as to minimize loss
        """
        # Different for hierarchical

        y_pred = self.predict_proba(X, **kwargs)
        if self.cost == "mse":
            loss = square_loss(y, y_pred)
            # negates loss if return_loss is False so that larger values loss values
            # translate to smaller score values (bigger loss = worse score). If return_loss is True
            # Then loss is returned and below expression ends up being score=loss
            score = (-1 * (not (return_loss)) + return_loss % 2) * loss
        elif self.cost == "cross_entropy":
            # loss = cross_entropy(y, y_pred)
            # TODO assuming here index 1 corresponds to p(x)=1
            loss_fn = nn.BCELoss()
            if type(y) == torch.Tensor:
                loss = loss_fn(y_pred[:, 1], y.double())
            else:
                loss = loss_fn(torch.tensor(y_pred[:, 1]), torch.tensor(y).double())
            # eps = torch.finfo(float).eps
            # loss = -torch.sum(torch.column_stack((1 - y, y)) * torch.log(y_pred + eps))
            score = (-1 * (not (return_loss)) + return_loss % 2) * loss

        return score

    def _construct_layer_dict(self, layer_defintion):
        """Constructs layer dictionary from config (containing list(str/int)) or the standard layer structure using the same
        circuit in each layer for pooling and convolution

        Args:
            layer_defintion (dict(list(str/int)) or tuple(str,str)): TODO
        """
        if type(layer_defintion) == type(dict()):
            layer_dict = {}
            for layer_name, layer_params in layer_defintion.items():
                layer_order = layer_params[0]
                layer_fn_name = layer_params[1]
                circ_name = layer_params[2]
                wire_pattern = layer_params[3]
                layer_fn = getattr(circuit_presets, layer_fn_name, None)
                if not (wire_pattern == None):
                    # If wire pattern is specified then ignore layerfn
                    layer_fn = None
                circuit_fn = getattr(circuit_presets, circ_name)
                # TODO document this assumption
                layer_type, param_count = (
                    ("convolutional", CIRCUIT_OPTIONS[circ_name])
                    if layer_name[0].upper() == "C"
                    else ("pooling", POOLING_OPTIONS[circ_name])
                )

                layer_dict[layer_name] = Layer(
                    layer_fn,
                    circuit_fn,
                    layer_type,
                    param_count,
                    layer_order,
                    wire_pattern,
                )
            return layer_dict.copy()
        elif type(layer_defintion) == type(tuple()) and len(layer_defintion) == 3:
            """
            The following is the default structure, it can be manually constructed as follows:
            layer_dict = {
                    "c_1": Layer(c_1, getattr(circuit_presets, circ_name), "convolutional", circ_param_count, 0,),
                    "p_1": Layer(p_1, getattr(circuit_presets, pool_name),"pooling",pool_param_count,1,),
                    "c_2": Layer(c_2, getattr(circuit_presets, circ_name),"convolutional",circ_param_count,2,),
                    "p_2": Layer(p_2, getattr(circuit_presets, pool_name),"pooling",pool_param_count,3,),
                    "c_3": Layer(c_3, getattr(circuit_presets, circ_name),"convolutional",circ_param_count,4,),
                    "p_3": Layer(p_3, getattr(circuit_presets, pool_name),"pooling",pool_param_count,5,),
                }
            """
            # TODO maybe named tuple is better here
            circ_name = layer_defintion[0]
            pool_name = layer_defintion[1]
            wire_pattern_args = layer_defintion[2]
            wire_combos = get_qcnn_graphs(**wire_pattern_args)

            layer_dict = {}
            layer_index = 0
            for layer_name, wire_combo in wire_combos.items():
                layer_order = layer_index
                layer_type, prefix, param_count, circuit_fn_name = (
                    ("convolutional", "c", CIRCUIT_OPTIONS[circ_name], circ_name)
                    if layer_name[0].upper() == "C"
                    else ("pooling", "p", POOLING_OPTIONS[pool_name], pool_name)
                )
                # layer_fn_name = f"{prefix}_{int(np.ceil((layer_index+1)/2))}"
                # If wire pattern is empty then use default layer functions
                layer_dict[layer_name] = Layer(
                    None,
                    getattr(circuit_presets, circuit_fn_name),
                    layer_type,
                    param_count,
                    layer_order,
                    wire_combo,
                )
                layer_index = layer_index + 1
            return layer_dict.copy()
        else:
            raise NotImplementedError(
                f"There is no implementation that supports the provided layer defintion"
            )

    def _sort_layer_dict_by_order(self):
        """
        Sorts the layer dictionary by the order that's provided.
        """
        self.layer_dict_ = {
            layer_name: layer
            for layer_name, layer in sorted(
                self.layer_dict_.items(), key=lambda x: x[1].layer_order
            )
        }

    def _evaluate(self):
        i = 0
        n = -1  # -1 means don't limit fn executions
        for layer_name, layer in self.layer_dict_.items():
            if (n == -1) or (i < n):
                if layer.layer_fn == None:
                    for wire_con in layer.wire_pattern:
                        layer.circuit(
                            self.coef_[self.coef_indices_[layer_name]], wire_con
                        )

                else:
                    layer.layer_fn(
                        layer.circuit, self.coef_[self.coef_indices_[layer_name]]
                    )
                # If modelling noise is required
                if self.noise:
                    # if layers are defined through wire combos function rather than custom structure
                    if (
                        type(self.layer_defintion) == type(tuple())
                        and len(self.layer_defintion) == 3
                    ):
                        # then the number of wires is contained in the layer definition
                        n_wires = self.layer_defintion[2]["n_wires"]
                    else:
                        # else assume 8 Qbits for now
                        n_wires = 8
                    for q in range(n_wires):
                        qml.DepolarizingChannel(self.noise, wires=q)
                        # qml.Depolarize(self.noise, wires=q)
            i = i + 1

    def _get_coef_information(self):
        total_coef_count = 0
        coef_indices = {}
        # Determine paramater indices per layer
        for layer_name, layer in self.layer_dict_.items():
            coef_indices[layer_name] = range(
                total_coef_count, total_coef_count + layer.param_count
            )
            total_coef_count += layer.param_count
        return total_coef_count, coef_indices.copy()


class Layer:
    """
    A generic layer consisting of some combination of variational circuits.
    Order doesn't have to be from 0, all layers get sorted purely by order value
    """

    def __init__(
        self, layer_fn, circuit, layer_type, param_count, layer_order, wire_pattern
    ):
        self.layer_fn = layer_fn
        self.circuit = circuit
        self.layer_type = layer_type
        self.param_count = param_count
        self.layer_order = layer_order
        self.wire_pattern = wire_pattern


# TODO difference between mixed and qubit for noise modelling?
# DEVICE = qml.device("default.qubit", wires=8)
# DEVICE = qml.device('qulacs.simulator', wires=8)
DEVICE = qml.device("default.qubit", wires=8)
# provider = IBMQ.load_account()


@qml.qnode(DEVICE, interface="torch")
def quantum_node(X, classifier):
    if getattr(classifier, "numpy", False):
        # If classifier needs to be deserialized
        classifier = classifier.numpy()
    if classifier.encoding_type:
        # If encoding is specified, otherwise don't encode
        apply_encoding(
            X,
            encoding_type=classifier.encoding_type,
            encoding_kwargs=classifier.encoding_kwargs,
        )
    classifier._evaluate()
    if classifier.cost == "mse":
        result = qml.expval(qml.PauliZ(classifier.response_wire_))
    elif classifier.cost == "cross_entropy":
        result = qml.probs(wires=classifier.response_wire_)
    return result


def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss


def cross_entropy(labels, predictions):
    # from sklearn.metrics import log_loss
    # log_loss(labels, [p._value for p in predictions])
    # TODO use pos_class implement ovr and then ensure labels is in a standardized format
    loss = 0
    for l, p in zip(labels, predictions):
        c_entropy = l * (anp.log(p[1])) + (1 - l) * anp.log(1 - p[1 - l])
        loss = loss + c_entropy

    return -1 * loss
