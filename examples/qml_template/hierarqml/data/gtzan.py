from hierarqml.data.base_data_module import BaseDataModule
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from hierarqml.utils import ParseAction
from collections import namedtuple
import numpy as np
import torch

Samples = namedtuple(
    "samples", ["x_train", "x_val", "x_test", "y_train", "y_val", "y_test"]
)

DEFAULT_GENRES = ("classical", "rock")


class GTZAN(BaseDataModule):
    """GTZAN dataset"""

    def __init__(self, args=None):
        super().__init__(args)
        self.pca_n_components = self.args.get("pca", None)
        self.data_path = self.args.get("data_path", None)
        self.genres = self.args.get("genres", DEFAULT_GENRES)

        self.input_dims = 57

        if self.pca_n_components:
            self.input_dims = self.pca_n_components

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument(
            "--pca",
            type=int,
            default=None,
            help=f"Number of components for PCA.",
        )
        parser.add_argument(
            "--data_path",
            type=str,
            default=None,
            help=f"Path of the GTZAN data file (csv).",
        )
        parser.add_argument("--genres", action=ParseAction, default=DEFAULT_GENRES)
        return parser

    def prepare_data(self, *args, **kwargs):
        if self.data_path is None:
            raise ValueError("Dataset path is None")

        data = pd.read_csv(self.data_path)

        # remove filename and length columns
        data = data.drop(columns=["filename", "length"])
        data = data[data["label"].isin(self.genres)]  # filter data

        # set label to 0 or 1
        data["label"] = data["label"].map({self.genres[0]: 0, self.genres[1]: 1})

        target = "label"  # specify target and features
        X, y = data.drop(columns=[target]), data[target]
        X, y = X.to_numpy(), y.to_numpy()

        # train, val and test split (70-15-15)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
        )

        self.samples = Samples(X_train, X_val, X_test, y_train, y_val, y_test)

    def setup(self, stage=None):

        steps = [
            ("mm_scaler", preprocessing.MinMaxScaler(feature_range=(0, np.pi / 2)))
        ]
        if self.pca_n_components:
            steps.append(("pca", PCA(self.pca_n_components)))

        # Create pipeline
        self.pipeline = Pipeline(steps)

        # Preprocess samples
        self.samples = Samples(
            self.pipeline.fit_transform(self.samples.x_train, self.samples.y_train),
            self.pipeline.transform(self.samples.x_val),
            self.pipeline.transform(self.samples.x_test),
            self.samples.y_train,
            self.samples.y_val,
            self.samples.y_test,
        )

        # Create datasets
        self.data_train = TensorDataset(
            torch.Tensor(self.samples.x_train), torch.LongTensor(self.samples.y_train)
        )
        self.data_val = TensorDataset(
            torch.Tensor(self.samples.x_val), torch.LongTensor(self.samples.y_val)
        )
        self.data_test = TensorDataset(
            torch.Tensor(self.samples.x_test), torch.LongTensor(self.samples.y_test)
        )
