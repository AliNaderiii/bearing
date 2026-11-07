"""Classification metrics incl. calibration."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score


def ece_score(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def compute_metrics(probs: np.ndarray, y: np.ndarray, n_classes: int) -> Dict:
    pred = probs.argmax(1)
    eps = 1e-12
    return {
        "acc": float((pred == y).mean() * 100),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(n_classes)), zero_division=0) * 100),
        "bal_acc": float(balanced_accuracy_score(y, pred) * 100),
        "nll": float(-np.log(probs[np.arange(len(y)), y] + eps).mean()),
        "ece": float(ece_score(probs, y) * 100),
        "n": int(len(y)),
        "confusion": confusion_matrix(y, pred, labels=list(range(n_classes))).tolist(),
    }
