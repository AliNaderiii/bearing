"""Model registry."""
from __future__ import annotations

from typing import Dict

import torch.nn as nn

from .baselines import DRSN, QCNN, WDCNN, ClassBD, CNNTransformer, StatsMLP
from .phyq_paper import PhyQTransNet
from .phyqnet import PhyQNet

REGISTRY = {
    "phyq_paper": PhyQTransNet,      # the author's model (PhyQ-TransNet, original architecture)
    "phyqnet": PhyQNet,
    "wdcnn": WDCNN,
    "qcnn": QCNN,
    "drsn": DRSN,
    "cnn_transformer": CNNTransformer,
    "stats_mlp": StatsMLP,
    "classbd": ClassBD,            # Liao et al., MSSP 2025 (re-implementation)
}


def build_model(mcfg: Dict, in_ch: int, n_classes: int, length: int) -> nn.Module:
    name = mcfg["name"].lower()
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](in_ch, n_classes, mcfg, length)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
