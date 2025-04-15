import argparse
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class QCNN(nn.Module):
    """QCNN using HierarQcal for classification"""

    def __init__(self, qnode, weight_shapes, args: argparse.Namespace = None):
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, x):
        return self.qlayer(x)

    @staticmethod
    def add_to_argparse(parser):
        return parser
