"""Base DataModule class."""

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import argparse
import os

BATCH_SIZE = 128
NUM_AVAIL_CPUS = len(os.sched_getaffinity(0))
DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS


class BaseDataModule(pl.LightningDataModule):
    """Base for all of our LightningDataModules."""

    def __init__(self, args: argparse.Namespace = None):
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", DEFAULT_NUM_WORKERS)
        self.input_dims: Tuple[int, ...]

        # Make sure to set the variables below in subclasses
        self.data_train: Dataset
        self.data_val: Dataset
        self.data_test: Dataset

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size",
            type=int,
            default=BATCH_SIZE,
            help=f"Number of examples to operate on per forward step. Default is {BATCH_SIZE}.",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=DEFAULT_NUM_WORKERS,
            help=f"Number of additional processes to load data. Default is {DEFAULT_NUM_WORKERS}.",
        )
        return parser

    def config(self):
        """Return important settings of the dataset, which will be passed to instantiate models."""
        return {"input_dims": self.input_dims}

    def prepare_data(self, *args, **kwargs):
        """Take the first steps to prepare data for use."""

    def setup(self, stage=None):
        """Perform final setup to prepare data for consumption by DataLoader."""

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
