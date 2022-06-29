# base
import importlib
import time
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class Qcnn_Classifier(BaseEstimator, ClassifierMixin):
    """
    from qcnn_estimator import Qcnn_Classifier
    from sklearn.utils.estimator_checks import check_estimator
    estimator = Qcnn_Classifier()
    check_estimator(estimator)
    """

    def __init__(
        self,
        quantum_fn_mapping,
        n_iter=25,
        learning_rate=0.01,
        batch_size=25,
        optimizer="adam",
        cost="cross_entropy",
        s_c=1,
        pool_filter="right",
        s_p=0,
        seed=1,
    ):
        self.quantum_fn_mapping = quantum_fn_mapping
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.cost = cost
        self.s_c = s_c
        self.pool_filter = pool_filter
        self.s_p = s_p
        self.seed = seed

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

        # - Quantum setup -
        self.quantum_module_ = importlib.import_module(
            self.quantum_fn_mapping["module_import_name"]
        )
        # Get quantum execution function
        self.quantum_node_ = getattr(self.quantum_module_, "quantum_execution")
        # Construct QCNN structure
        self.layer_dict_ = self._construct_layer_dict()
        # Sort the layers according to the order provided
        self._sort_layer_dict_by_order()
        # Get coefficient information
        self.coef_count_, self.coef_indices_ = self._get_coef_information()
        # Initialize Coefficients
        # if self.seed:
        #     torch.manual_seed(self.seed) TODO way to set seed
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
            y_hat = [self.quantum_node_(x, self) for x in X]
            y_hat = torch.stack(y_hat)
            # parallel
            # pool = multiprocessing.Pool()
            # # y_hat = pool.map(partial(qcnn_estimator.quantum_node,classifier=self), X)
            # y_hat = Parallel(n_jobs=2)(
            #     delayed(partial(quantum_node, classifier=self))(x) for x in X
            # )
        else:
            y_hat = np.array([quantum_node_(x, self).numpy() for x in X])
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

    def _construct_layer_dict(self):
        """Constructs layer dictionary from config (containing list(str/int)) or the standard layer structure using the same
        circuit in each layer for pooling and convolution

        Args:self, unit_fn, param_count, wire_pattern, block_order, predefined_structure=None
            layer_defintion (dict(list(str/int)) or tuple(str,str)): TODO
        """
        self.qcnn_graphs_ = self._get_qcnn_graphs(
            self.n_wires, self.s_c, self.pool_filter, self.s_p
        )
        layer_dict = {}
        for layer, graph in self.qcnn_graphs.items():
            # Notational scheme is layer -> C graph --> Qc,Ec, P Graph --> Qp,Ep
            E_cl = graph[layer][0][1]
            E_pl = graph[layer][1][1]
            (
                (convolution_circuit, c_param_count),
                (pooling_circuit, p_parm_count),
            ) = self.quantum_fn_mapping["layer_fn_mapping"].get(
                layer, 1
            )  # default to layer 1 if specific layer is not defined TODO mention in docs that atleast layer 1 is required

            # layer_fn_name = f"{prefix}_{int(np.ceil((layer_index+1)/2))}"
            layer_dict[f"c_{layer}"] = Block(
                getattr(self.quantum_module_, convolution_circuit),
                c_param_count,
                E_cl,
                2 * layer - 1,
            )
            layer_dict[f"c_{layer}"] = Block(
                getattr(self.quantum_module_, pooling_circuit),
                p_parm_count,
                E_pl,
                2 * layer,
            )
        return layer_dict.copy()

    def _get_qcnn_graphs(n_wires, s_c, pool_filter, s_p=0):
        """ """
        if type(pool_filter) is str:
            # Mapping words to the filter type
            if pool_filter == "left":
                # 0 1 2 3 4 5 6 7
                # x x x x
                pool_filter = lambda arr: arr[0 : len(arr) // 2 : 1]
            elif pool_filter == "right":
                # 0 1 2 3 4 5 6 7
                #         x x x x
                pool_filter = lambda arr: arr[len(arr) : len(arr) // 2 - 1 : -1]
            elif pool_filter == "eo_even":
                # 0 1 2 3 4 5 6 7
                # x   x   x   x
                pool_filter = lambda arr: arr[0::2]
            elif pool_filter == "eo_odd":
                # 0 1 2 3 4 5 6 7
                #   x   x   x   x
                pool_filter = lambda arr: arr[1::2]
            elif pool_filter == "inside":
                # 0 1 2 3 4 5 6 7
                #     x x x x
                pool_filter = lambda arr: arr[
                    len(arr) // 2 - len(arr) // 4 : len(arr) // 2 + len(arr) // 4 : 1
                ]  # inside
            elif pool_filter == "outside":
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

        graphs = {}
        layer = 1
        Qc_l = [i + 1 for i in range(n_wires)]  # We label the nodes from 1 to n
        Qp_l = Qc_l.copy()
        while len(Qc_l) > 1:

            nq_avaiable = len(Qc_l)
            mod_nq = lambda x: x % nq_avaiable
            Ec_l = [(Qc_l[i], Qc_l[mod_nq(i + s_c)]) for i in range(nq_avaiable)]
            if len(Ec_l) == 2 and Ec_l[0][0:] == Ec_l[1][1::-1]:
                # TODO improve this, the issue is (1,2) and (2,1) with this logic, there might be a better
                # TODO way to traverse the graph in a general way
                Ec_l = [Ec_l[0]]
            measured_q = pool_filter(Qc_l)
            remaining_q = [q for q in Qc_l if not (q in measured_q)]
            Ep_l = [
                (measured_q[i], remaining_q[(i + s_p) % len(remaining_q)])
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

    def _sort_layer_dict_by_order(self):
        """
        Sorts the layer dictionary by the order that's provided.
        """
        self.layer_dict_ = {
            layer_name: layer
            for layer_name, layer in sorted(
                self.layer_dict_.items(), key=lambda x: x[1].block_order
            )
        }

    def _evaluate(self):
        for block_name, block in self.layer_dict_.items():
            if block.predefined_structure == None:
                for wire_con in block.wire_pattern:
                    block.unit_fn(self.coef_[self.coef_indices_[block_name]], wire_con)
            else:
                block.predefined_structure(
                    block.unit_fn, self.coef_[self.coef_indices_[block_name]]
                )

    def _get_coef_information(self):
        total_coef_count = 0
        coef_indices = {}
        # Determine paramater indices per layer
        for layer_name, block in self.layer_dict_.items():
            coef_indices[layer_name] = range(
                total_coef_count, total_coef_count + block.param_count
            )
            total_coef_count += block.param_count
        return total_coef_count, coef_indices.copy()


class Block:
    """
    A generic circuit block consisting of some combination of variational circuits.
    Order doesn't have to be from 0, all layers get sorted purely by order value
    """

    def __init__(
        self, unit_fn, param_count, wire_pattern, block_order, predefined_structure=None
    ):
        self.unit_fn = unit_fn
        self.param_count = param_count
        self.wire_pattern = wire_pattern
        self.block_order = block_order
        self.predefined_structure = predefined_structure
