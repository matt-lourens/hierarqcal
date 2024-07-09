"""Experiment-running framework."""

import argparse
import numpy as np
import pytorch_lightning as pl
import torch

from hierarqml.models import BaseLitModel
from training.util import (
    DATA_CLASS_MODULE,
    import_class,
    MODEL_CLASS_MODULE,
    setup_data_and_model_from_args,
    embd_add_to_argparse,
)

# In order to ensure reproducible experiments, we must set random seeds.
np.random.seed(42)
torch.manual_seed(42)


def _setup_parser():

    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])
    parser.set_defaults(max_epochs=1)

    # Basic arguments
    parser.add_argument(
        "--data_class",
        type=str,
        default="GTZAN",
        help=f"String identifier for the data class, relative to {DATA_CLASS_MODULE}.",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="QCNN",
        help=f"String identifier for the model class, relative to {MODEL_CLASS_MODULE}.",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default="AngleEmbedding",
        choices=["AngleEmbedding", "AmplitudeEmbedding"],
        help="String identifier for the Quantum Embedding.",
    )
    parser.add_argument(
        "--ansatz",
        type=str,
        default="./training/ansatz",
        help="String identifier of the HierarQcal Ansatz",
    )

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = import_class(f"{DATA_CLASS_MODULE}.{temp_args.data_class}")
    model_class = import_class(f"{MODEL_CLASS_MODULE}.{temp_args.model_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    BaseLitModel.add_to_argparse(lit_model_group)

    # Get the embedding classes and add its specific arguments
    embd_class = temp_args.embedding
    embd_group = parser.add_argument_group("Embedding Args")
    embd_add_to_argparse(embd_class, embd_group)

    parser.add_argument("--help", "-h", action="help")
    return parser


def main():
    parser = _setup_parser()
    args = parser.parse_args()

    data, model = setup_data_and_model_from_args(args)
    lit_model = BaseLitModel(model, args)

    logger = pl.loggers.CSVLogger(save_dir="csv_logs")

    trainer = pl.Trainer.from_argparse_args(args, logger=logger)
    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data)


if __name__ == "__main__":
    main()
