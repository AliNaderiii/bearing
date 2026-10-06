# -*- coding: utf-8 -*-
r"""
sanity_check_pu_npz_vscode.py
Author: Ali Naderi + ChatGPT (Q1 Sanity-Check | VSCode-Friendly Edition)

✔ Works with NO command-line arguments (VSCode Run-friendly)
✔ Still supports full CLI usage
✔ Leakage, distribution, signal, feature & baseline checks
"""

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ======================================================
# Default paths (USED WHEN NO CLI ARGS ARE GIVEN)
# ======================================================
DEFAULT_NPZ = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\Paderborn\processed_pu_full.npz"


# ======================================================
# Utilities
# ======================================================
def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def safe_json_load(x):
    if isinstance(x, np.ndarray) and x.dtype == object:
        x = x.item()
    if isinstance(x, (bytes, bytearray)):
        x = x.decode("utf-8", errors="ignore")
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return {}
    return {}


def bincount(y: np.ndarray, n: int):
    return np.bincount(y.astype(np.int64), minlength=n)


# ======================================================
# Argument parser (VSCode-safe)
# ======================================================
def parse_args():
    parser = argparse.ArgumentParser("PU NPZ Sanity Check (VSCode Friendly)")
    parser.add_argument("--npz", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--svm_rbf", action="store_true")

    args, unknown = parser.parse_known_args()

    # ---- VSCode default handling ----
    if not args.npz:
        args.npz = DEFAULT_NPZ

    if not args.out_dir:
        args.out_dir = str(Path(args.npz).with_suffix("")) + "_sanity_report"

    return args


# ======================================================
# Main
# ======================================================
def main():
    args = parse_args()

    npz_path = Path(args.npz).resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    out_dir = ensure_dir(args.out_dir)
    fig_dir = ensure_dir(os.path.join(out_dir, "figures"))

    print("=" * 60)
    print("PU SANITY CHECK")
    print(f"NPZ     : {npz_path}")
    print(f"OUT_DIR : {out_dir}")
    print("=" * 60)

    data = np.load(npz_path, allow_pickle=True)

    # ---------------- Load essentials ----------------
    Xtr = data["X_train_sig"]
    Xva = data["X_val_sig"]
    Xte = data["X_test_sig"]

    ytr = data["y_train"]
    yva = data["y_val"]
    yte = data["y_test"]

    classes = [str(c) for c in data["classes"].tolist()]
    n_classes = len(classes)

    manifest = safe_json_load(data["manifest_json"])

    # ---------------- Basic checks ----------------
    assert Xtr.ndim == 3 and Xva.ndim == 3 and Xte.ndim == 3
    assert Xtr.shape[1:] == Xva.shape[1:] == Xte.shape[1:]

    print(f"Signals shape: {Xtr.shape[1]} channels × {Xtr.shape[2]} length")

    # ---------------- Class distributions ----------------
    tr_dist = bincount(ytr, n_classes)
    va_dist = bincount(yva, n_classes)
    te_dist = bincount(yte, n_classes)

    print("Train dist:", dict(zip(classes, tr_dist)))
    print("Val   dist:", dict(zip(classes, va_dist)))
    print("Test  dist:", dict(zip(classes, te_dist)))

    # ---------------- Leakage check ----------------
    groups = manifest.get("groups", {})
    tr_g = set(groups.get("train", []))
    va_g = set(groups.get("val", []))
    te_g = set(groups.get("test", []))

    leakage_ok = not (tr_g & va_g or tr_g & te_g or va_g & te_g)
    print("Group leakage check:", "OK" if leakage_ok else "FAIL")

    # ---------------- Plot distributions ----------------
    def plot_dist(dist, title, fname):
        plt.figure()
        plt.bar(range(len(classes)), dist)
        plt.xticks(range(len(classes)), classes, rotation=20)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, fname), dpi=300)
        plt.close()

    plot_dist(tr_dist, "Train class distribution", "train_dist.png")
    plot_dist(va_dist, "Val class distribution", "val_dist.png")
    plot_dist(te_dist, "Test class distribution", "test_dist.png")

    # ---------------- Waveform sanity ----------------
    plt.figure(figsize=(10, 4))
    plt.plot(Xtr[0, 0, :])
    plt.title("Example waveform (Train, ch0)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "example_waveform.png"), dpi=300)
    plt.close()

    # ---------------- Feature baseline ----------------
    if "X_train_feat" in data.files:
        print("Running baseline on engineered features...")

        Xtrf = data["X_train_feat"]
        Xvaf = data["X_val_feat"]
        Xtef = data["X_test_feat"]

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                n_jobs=None
            ))
        ])

        clf.fit(Xtrf, ytr)

        for split, X, y in [
            ("Train", Xtrf, ytr),
            ("Val", Xvaf, yva),
            ("Test", Xtef, yte),
        ]:
            pred = clf.predict(X)
            acc = accuracy_score(y, pred)
            f1m = f1_score(y, pred, average="macro")
            print(f"[LR] {split}: acc={acc:.4f} f1_macro={f1m:.4f}")

        cm = confusion_matrix(yte, clf.predict(Xtef))
        plt.figure(figsize=(6, 5))
        plt.imshow(cm)
        plt.xticks(range(n_classes), classes, rotation=30)
        plt.yticks(range(n_classes), classes)
        plt.title("Confusion Matrix (LR – Test)")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "cm_lr_test.png"), dpi=300)
        plt.close()

    # ---------------- Save report ----------------
    report = {
        "npz": str(npz_path),
        "classes": classes,
        "train_dist": tr_dist.tolist(),
        "val_dist": va_dist.tolist(),
        "test_dist": te_dist.tolist(),
        "leakage_free": leakage_ok,
        "signal_shape": Xtr.shape[1:],
    }

    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print("SANITY CHECK DONE")
    print(f"Figures saved to: {fig_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
