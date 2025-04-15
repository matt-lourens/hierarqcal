import importlib
from hierarqml.utils import get_circuit
import argparse
import inspect
import pennylane as qml
import numpy as np

DATA_CLASS_MODULE = "hierarqml.data"
MODEL_CLASS_MODULE = "hierarqml.models"
EMBEDDING_FEATURES_EXCLUDED = ["features", "wires", "id"]


def import_class(module_and_class_name: str) -> type:
    """Import class from a module."""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def setup_data_and_model_from_args(args: argparse.Namespace):
    data_class = import_class(f"{DATA_CLASS_MODULE}.{args.data_class}")
    model_class = import_class(f"{MODEL_CLASS_MODULE}.{args.model_class}")

    data = data_class(args)

    # Creating Quantum Circuit using HierarQcal and PennyLane

    # Custom HierarQcal ansatz from file
    # the motif should be defined in a function named `get_motif`
    file_path = f"{args.ansatz}.py"
    module_name = file_path.split("/")[-1].replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    imported_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imported_module)
    get_motif = imported_module.get_motif

    n = data.config()["input_dims"]
    if "Amplitude" in args.embedding:
        n = int(np.ceil(np.log2(n)))

    # Get embedding signature values from args
    sig = inspect.signature(getattr(qml, args.embedding))
    sig = list(sig.parameters.keys())
    kwargs = {
        key: getattr(args, key) for key in sig if key not in EMBEDDING_FEATURES_EXCLUDED
    }

    # Motif, QNode and shape of weights
    # Needed for the Quantum Model
    motif = get_motif(n)
    qnode = get_circuit(motif, embedding=args.embedding, **kwargs)
    weight_shapes = {"weights": motif.n_symbols}

    model = model_class(qnode, weight_shapes, args=args)

    return data, model


def embd_add_to_argparse(embedding, parser):
    """Adds the signature of the given embedding to parser"""

    # Default values of the signature
    embd_default_map = {"pad_with": 0.0, "normalize": True, "rotation": "X,"}

    sig = inspect.signature(getattr(qml, embedding))
    sig = list(sig.parameters.keys())

    for key in sig:
        if key not in EMBEDDING_FEATURES_EXCLUDED:
            parser.add_argument(
                f"--{key}",
                type=type(embd_default_map[key]),
                default=embd_default_map[key],
            )

    return parser
