# -*- coding: utf-8 -*-
r"""
train_ims.py
Author: Ali Naderi - 2026 (Final + Publication Edition)

Purpose:
  Train your custom CWRU-designed model (e.g., QDCA_Hybrid_Pro) on IMS dataset NPZ output.

NPZ Expected Keys:
  X_train_sig, X_val_sig, X_test_sig : (N, C, L) float32
  X_train_feat, X_val_feat, X_test_feat : (N, F) float32   [optional but recommended]
  y_train, y_val, y_test : (N,) int64
  classes : object/list/ndarray  [optional]
  scaler_mean, scaler_scale : (F,) float32  [optional; for feature normalization]
  manifest_json : bytes/str      [optional]
  test_snr_-10_sig, test_snr_-10_feat, ...  [optional robustness test sets]

Model Forward:
  forward(x) or forward(x, features=None)

Usage (CLI):
  python train_ims.py --npz "D:\Bearing Project\4. Bearings\IMS\processed_ims\processed_ims_full.npz" ^
    --out_dir "D:\Bearing Project\4. Bearings\IMS\runs_ims" ^
    --seeds 42 101 2024 ^
    --epochs 60 --batch_size 64 --lr 1e-3 --scheduler cosine ^
    --use_learned_filters ^
    --filter_ch0 "D:\Bearing Project\4. Bearings\IMS\processed_ims\learned_filter_ims_ch0.pt" ^
    --filter_ch1 "D:\Bearing Project\4. Bearings\IMS\processed_ims\learned_filter_ims_ch1.pt" ^
    --filter_ch2 "D:\Bearing Project\4. Bearings\IMS\processed_ims\learned_filter_ims_ch2.pt"

VSCode tip:
  If you run without args, defaults are auto-injected (edit them below in main()).
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import argparse
import random
import datetime
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# matplotlib only (publication-safe)
import matplotlib.pyplot as plt

# sklearn metrics for paper figures
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)

# -----------------------------
# Logging
# -----------------------------
LOG = logging.getLogger("train_ims")
LOG.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
LOG.handlers.clear()
LOG.addHandler(_handler)


# -----------------------------
# Reproducibility
# -----------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Utilities
# -----------------------------
def now_stamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_dir(p: str | Path) -> str:
    p = str(p)
    os.makedirs(p, exist_ok=True)
    return p


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_npz(path: str) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def infer_num_classes(y: np.ndarray, classes_obj: Any) -> int:
    if classes_obj is None:
        return int(np.max(y)) + 1
    try:
        if isinstance(classes_obj, np.ndarray) and classes_obj.shape == ():
            classes_obj = classes_obj.item()
        if isinstance(classes_obj, (list, tuple)):
            return len(classes_obj)
        if isinstance(classes_obj, np.ndarray):
            return len(classes_obj)
    except Exception:
        pass
    return int(np.max(y)) + 1


def get_classes_list(classes_obj: Any) -> Optional[List[str]]:
    try:
        if classes_obj is None:
            return None
        if isinstance(classes_obj, np.ndarray) and classes_obj.shape == ():
            classes_obj = classes_obj.item()
        if isinstance(classes_obj, (list, tuple)):
            return [str(x) for x in classes_obj]
        if isinstance(classes_obj, np.ndarray):
            return [str(x) for x in classes_obj.tolist()]
        return [str(x) for x in classes_obj]
    except Exception:
        return None


def safe_decode_manifest(mj: Any) -> Optional[str]:
    try:
        if mj is None:
            return None
        if isinstance(mj, np.ndarray) and mj.shape == ():
            mj = mj.item()
        if isinstance(mj, (bytes, bytearray)):
            return mj.decode("utf-8", errors="ignore")
        return str(mj)
    except Exception:
        return None


# -----------------------------
# Dynamic model import
# -----------------------------
def import_model_module(model_py: str) -> Any:
    import importlib.util
    model_py = os.path.abspath(model_py)
    spec = importlib.util.spec_from_file_location("user_model_module", model_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load model module from: {model_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pick_model_class(module: Any, preferred: Optional[str] = None) -> type:
    if preferred and hasattr(module, preferred):
        cls = getattr(module, preferred)
        if isinstance(cls, type) and issubclass(cls, nn.Module):
            return cls

    for name in ["QDCA_Hybrid_Pro", "QDCA_Hybrid", "QDCA_Net", "Model", "Net"]:
        if hasattr(module, name):
            cls = getattr(module, name)
            if isinstance(cls, type) and issubclass(cls, nn.Module):
                return cls

    for _, obj in vars(module).items():
        if isinstance(obj, type) and issubclass(obj, nn.Module):
            return obj

    raise RuntimeError("No nn.Module subclass found in model module.")


# -----------------------------
# Learned filter application
# -----------------------------
@torch.no_grad()
def apply_learned_filters(
    x: torch.Tensor,
    filt_ch0: Optional[torch.Tensor],
    filt_ch1: Optional[torch.Tensor],
    filt_ch2: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    x: (B, C, L)
    filters: FIR (K,) or compatible shapes -> applied per-channel conv1d SAME
    """
    B, C, L = x.shape
    if C != 3:
        return x

    filts = [filt_ch0, filt_ch1, filt_ch2]
    out = []
    for c in range(3):
        f = filts[c]
        if f is None:
            out.append(x[:, c:c+1, :])
            continue
        f = f.detach().float().to(x.device).view(-1)
        K = int(f.numel())
        w = f.view(1, 1, K)
        pad = K // 2
        yc = F.conv1d(x[:, c:c+1, :], w, bias=None, stride=1, padding=pad)
        if yc.shape[-1] > L:
            yc = yc[..., :L]
        out.append(yc)
    return torch.cat(out, dim=1)


def load_filter(path: Optional[str]) -> Optional[torch.Tensor]:
    if not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Filter not found: {path}")

    # safer default for future torch: weights_only where possible
    try:
        t = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        t = torch.load(path, map_location="cpu")

    if isinstance(t, dict):
        for k in ["filter", "weight", "w", "kernel"]:
            if k in t and torch.is_tensor(t[k]):
                t = t[k]
                break
        else:
            for v in t.values():
                if torch.is_tensor(v):
                    t = v
                    break

    if not torch.is_tensor(t):
        raise RuntimeError(f"Loaded filter is not a tensor from: {path}")
    return t


# -----------------------------
# Loss: Label smoothing
# -----------------------------
class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.0):
        super().__init__()
        self.smoothing = float(max(0.0, min(0.5, smoothing)))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.smoothing <= 0.0:
            return F.cross_entropy(logits, target)
        n_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


# -----------------------------
# cuDNN FIND fallback
# -----------------------------
def enable_stable_cudnn(disable_cudnn: bool) -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if disable_cudnn:
        torch.backends.cudnn.enabled = False
        LOG.warning("cuDNN disabled by flag --disable_cudnn")


def should_fallback_disable_cudnn(err: Exception) -> bool:
    msg = str(err).lower()
    return ("find was unable to find an engine" in msg) or ("cudnn frontend" in msg) or ("no engine" in msg)


# -----------------------------
# Feature normalization (from NPZ scaler)
# -----------------------------
def apply_feature_scaler(
    X: np.ndarray,
    scaler_mean: Optional[np.ndarray],
    scaler_scale: Optional[np.ndarray],
) -> np.ndarray:
    if X is None:
        return X
    if scaler_mean is None or scaler_scale is None:
        return X
    try:
        m = np.array(scaler_mean, dtype=np.float32).reshape(1, -1)
        s = np.array(scaler_scale, dtype=np.float32).reshape(1, -1)
        s = np.where(s == 0, 1.0, s)
        return (X - m) / s
    except Exception as e:
        LOG.warning(f"Could not apply scaler normalization: {e}")
        return X


# -----------------------------
# Metrics + prediction
# -----------------------------
@torch.no_grad()
def predict_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_features: bool,
    use_learned_filters: bool,
    filt0: Optional[torch.Tensor],
    filt1: Optional[torch.Tensor],
    filt2: Optional[torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_y = []
    for batch in loader:
        if use_features:
            x, feat, y = batch
            x = x.to(device, non_blocking=True)
            feat = feat.to(device, non_blocking=True)
        else:
            x, y = batch
            x = x.to(device, non_blocking=True)
            feat = None
        y = y.to(device, non_blocking=True)

        if use_learned_filters:
            x = apply_learned_filters(x, filt0, filt1, filt2)

        logits = model(x, feat) if use_features else model(x)
        all_logits.append(logits.detach().float().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_y, axis=0)


@torch.no_grad()
def evaluate_loss_acc(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    use_features: bool,
    use_learned_filters: bool,
    filt0: Optional[torch.Tensor],
    filt1: Optional[torch.Tensor],
    filt2: Optional[torch.Tensor],
) -> Dict[str, float]:
    model.eval()
    losses = []
    correct = 0
    total = 0
    for batch in loader:
        if use_features:
            x, feat, y = batch
            x = x.to(device, non_blocking=True)
            feat = feat.to(device, non_blocking=True)
        else:
            x, y = batch
            x = x.to(device, non_blocking=True)
            feat = None
        y = y.to(device, non_blocking=True)

        if use_learned_filters:
            x = apply_learned_filters(x, filt0, filt1, filt2)

        logits = model(x, feat) if use_features else model(x)
        loss = criterion(logits, y)
        losses.append(float(loss.item()) * x.size(0))
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        total += int(x.size(0))

    loss_mean = float(sum(losses) / max(1, total))
    acc = float(correct / max(1, total))
    return {"loss": loss_mean, "acc": acc}


# -----------------------------
# Scheduler factory
# -----------------------------
def make_scheduler(optimizer: optim.Optimizer, args: argparse.Namespace):
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.scheduler == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    if args.scheduler == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=args.gamma, patience=args.patience,
            threshold=args.min_delta, verbose=True
        )
    raise ValueError(f"Unknown scheduler: {args.scheduler}")


# -----------------------------
# Early stopping
# -----------------------------
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = -1e18
        self.bad = 0

    def step(self, metric: float) -> bool:
        if metric > self.best + self.min_delta:
            self.best = metric
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience


# -----------------------------
# Plotting (publication figures)
# -----------------------------
def _savefig(path: str, dpi: int = 450) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_training_curves(history: List[Dict[str, Any]], out_dir: str) -> None:
    epochs = [h["epoch"] for h in history]
    tr_loss = [h["train_loss"] for h in history]
    va_loss = [h["val_loss"] for h in history]
    tr_acc = [h["train_acc"] for h in history]
    va_acc = [h["val_acc"] for h in history]

    plt.figure()
    plt.plot(epochs, tr_loss, label="Train Loss")
    plt.plot(epochs, va_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    _savefig(os.path.join(out_dir, "curve_loss.png"))

    plt.figure()
    plt.plot(epochs, tr_acc, label="Train Acc")
    plt.plot(epochs, va_acc, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    _savefig(os.path.join(out_dir, "curve_acc.png"))


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str, out_path: str) -> None:
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # annotate
    thresh = cm.max() * 0.6 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            plt.text(j, i, f"{v}", ha="center", va="center",
                     color="white" if v > thresh else "black")
    _savefig(out_path)


def plot_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], out_path: str) -> Dict[str, Any]:
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(class_names))), zero_division=0)

    x = np.arange(len(class_names))
    width = 0.25

    plt.figure()
    plt.bar(x - width, p, width, label="Precision")
    plt.bar(x, r, width, label="Recall")
    plt.bar(x + width, f1, width, label="F1")
    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.ylim(0, 1.02)
    plt.ylabel("Score")
    plt.title("Per-class Precision / Recall / F1")
    plt.legend()
    _savefig(out_path)

    return {
        "precision": p.tolist(),
        "recall": r.tolist(),
        "f1": f1.tolist(),
        "support": sup.tolist(),
    }


def plot_snr_curve(snr_acc: Dict[str, float], out_path: str) -> None:
    # sort numeric snrs, keep others last
    def _key(k: str):
        try:
            return (0, float(k))
        except Exception:
            return (1, k)

    keys = sorted(snr_acc.keys(), key=_key)
    xs = []
    ys = []
    for k in keys:
        xs.append(k)
        ys.append(snr_acc[k])

    plt.figure()
    plt.plot(range(len(xs)), ys, marker="o")
    plt.xticks(range(len(xs)), xs)
    plt.ylim(0, 1.02)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.title("Robustness Curve (SNR vs Accuracy)")
    _savefig(out_path)


def plot_seed_summary_bar(mean_acc: float, std_acc: float, seeds: List[int], out_path: str) -> None:
    plt.figure()
    plt.bar([0], [mean_acc], yerr=[std_acc], capsize=10)
    plt.xticks([0], [f"TEST\n(seeds={seeds})"])
    plt.ylim(0, 1.02)
    plt.ylabel("Accuracy")
    plt.title("Mean ± Std Accuracy over Seeds")
    _savefig(out_path)


# -----------------------------
# Train step (AMP updated)
# -----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
    grad_clip: float,
    use_amp: bool,
    use_features: bool,
    use_learned_filters: bool,
    filt0: Optional[torch.Tensor],
    filt1: Optional[torch.Tensor],
    filt2: Optional[torch.Tensor],
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    for batch in loader:
        if use_features:
            x, feat, y = batch
            x = x.to(device, non_blocking=True)
            feat = feat.to(device, non_blocking=True)
        else:
            x, y = batch
            x = x.to(device, non_blocking=True)
            feat = None
        y = y.to(device, non_blocking=True)

        if use_learned_filters:
            x = apply_learned_filters(x, filt0, filt1, filt2)

        optimizer.zero_grad(set_to_none=True)

        try:
            with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                logits = model(x, feat) if use_features else model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        except RuntimeError as e:
            if (x.is_cuda and torch.backends.cudnn.enabled and should_fallback_disable_cudnn(e)):
                LOG.error("cuDNN FIND engine error detected. Disabling cuDNN and retrying batch (no AMP)...")
                torch.backends.cudnn.enabled = False
                optimizer.zero_grad(set_to_none=True)

                logits = model(x, feat) if use_features else model(x)
                loss = criterion(logits, y)

                loss.backward()
                if grad_clip and grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            else:
                raise

        bs = int(x.size(0))
        total_loss += float(loss.item()) * bs
        pred = torch.argmax(logits.detach(), dim=1)
        correct += int((pred == y).sum().item())
        total += bs

    return {"loss": float(total_loss / max(1, total)), "acc": float(correct / max(1, total))}


# -----------------------------
# Argparser
# -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train your CWRU-designed model on IMS NPZ (multi-seed + paper plots).")

    p.add_argument("--npz", type=str, required=True, help="Path to processed_ims_full.npz")
    p.add_argument("--out_dir", type=str, required=True, help="Output root directory (runs created inside).")

    p.add_argument("--model_py", type=str, default=None,
                   help="Path to your model.py. If omitted, uses ./Model/model.py next to this script.")
    p.add_argument("--model_class", type=str, default=None,
                   help="Optional class name inside model.py (e.g., QDCA_Hybrid_Pro).")

    p.add_argument("--seed", type=int, default=42, help="Fallback single seed")
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="Multi-seed list. Example: --seeds 42 101 2024")

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--label_smoothing", type=float, default=0.0)

    p.add_argument("--scheduler", type=str, default="cosine",
                   choices=["none", "cosine", "step", "plateau"])
    p.add_argument("--step_size", type=int, default=15)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min_delta", type=float, default=0.0)

    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--amp", action="store_true", help="Enable AMP (GPU).")

    p.add_argument("--use_learned_filters", action="store_true")
    p.add_argument("--filter_ch0", type=str, default=None)
    p.add_argument("--filter_ch1", type=str, default=None)
    p.add_argument("--filter_ch2", type=str, default=None)

    p.add_argument("--disable_cudnn", action="store_true")

    p.add_argument("--resume", type=str, default=None, help="Optional checkpoint to resume (per-seed).")
    return p


# -----------------------------
# Single run (one seed)
# -----------------------------
def run_one_seed(args: argparse.Namespace, seed: int) -> Dict[str, Any]:
    seed_everything(seed)
    enable_stable_cudnn(args.disable_cudnn)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    LOG.info(f"Device: {device}")

    if not os.path.isfile(args.npz):
        raise FileNotFoundError(f"NPZ not found: {args.npz}")

    data = load_npz(args.npz)

    # core arrays
    X_train_sig = data["X_train_sig"].astype(np.float32)
    X_val_sig   = data["X_val_sig"].astype(np.float32)
    X_test_sig  = data["X_test_sig"].astype(np.float32)

    y_train = data["y_train"].astype(np.int64)
    y_val   = data["y_val"].astype(np.int64)
    y_test  = data["y_test"].astype(np.int64)

    X_train_feat = data.get("X_train_feat", None)
    X_val_feat   = data.get("X_val_feat", None)
    X_test_feat  = data.get("X_test_feat", None)

    if X_train_feat is not None: X_train_feat = X_train_feat.astype(np.float32)
    if X_val_feat   is not None: X_val_feat   = X_val_feat.astype(np.float32)
    if X_test_feat  is not None: X_test_feat  = X_test_feat.astype(np.float32)

    scaler_mean = data.get("scaler_mean", None)
    scaler_scale = data.get("scaler_scale", None)

    classes_obj = data.get("classes", None)
    num_classes = infer_num_classes(y_train, classes_obj)
    classes_list = get_classes_list(classes_obj) or [str(i) for i in range(num_classes)]

    # warn missing classes in test
    uniq_test = set(np.unique(y_test).tolist())
    if len(uniq_test) < num_classes:
        missing = sorted(list(set(range(num_classes)) - uniq_test))
        LOG.warning(f"Test set is missing classes: {missing}. Metrics may be biased.")

    use_features = (X_train_feat is not None) and (X_val_feat is not None) and (X_test_feat is not None)
    if use_features:
        X_train_feat = apply_feature_scaler(X_train_feat, scaler_mean, scaler_scale)
        X_val_feat   = apply_feature_scaler(X_val_feat, scaler_mean, scaler_scale)
        X_test_feat  = apply_feature_scaler(X_test_feat, scaler_mean, scaler_scale)
        LOG.info("Applied feature normalization using scaler_mean/scaler_scale from NPZ.")

    # run folder
    run_dir = ensure_dir(os.path.join(args.out_dir, f"run_{now_stamp()}_seed{seed}"))
    LOG.info(f"OUT: {run_dir}")
    LOG.info(f"Train/Val/Test: {len(y_train)}/{len(y_val)}/{len(y_test)} | num_classes={num_classes}")

    # save config + manifest
    cfg = vars(args).copy()
    cfg["seed_used"] = seed
    save_json(cfg, os.path.join(run_dir, "config.json"))

    mj = safe_decode_manifest(data.get("manifest_json", None))
    if mj is not None:
        with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
            f.write(mj)

    # dataloaders
    pin = bool(args.pin_memory and device.type == "cuda")

    if use_features:
        ds_train = TensorDataset(torch.from_numpy(X_train_sig), torch.from_numpy(X_train_feat), torch.from_numpy(y_train))
        ds_val   = TensorDataset(torch.from_numpy(X_val_sig),   torch.from_numpy(X_val_feat),   torch.from_numpy(y_val))
        ds_test  = TensorDataset(torch.from_numpy(X_test_sig),  torch.from_numpy(X_test_feat),  torch.from_numpy(y_test))
    else:
        ds_train = TensorDataset(torch.from_numpy(X_train_sig), torch.from_numpy(y_train))
        ds_val   = TensorDataset(torch.from_numpy(X_val_sig),   torch.from_numpy(y_val))
        ds_test  = TensorDataset(torch.from_numpy(X_test_sig),  torch.from_numpy(y_test))

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=pin, drop_last=False)
    dl_val   = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=pin, drop_last=False)
    dl_test  = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=pin, drop_last=False)

    # learned filters
    filt0 = filt1 = filt2 = None
    if args.use_learned_filters:
        filt0 = load_filter(args.filter_ch0)
        filt1 = load_filter(args.filter_ch1)
        filt2 = load_filter(args.filter_ch2)
        torch.save(filt0, os.path.join(run_dir, "learned_filter_ch0.pt"))
        torch.save(filt1, os.path.join(run_dir, "learned_filter_ch1.pt"))
        torch.save(filt2, os.path.join(run_dir, "learned_filter_ch2.pt"))
        LOG.info("Loaded learned filters (ch0/ch1/ch2) and copied into run folder.")

    # model import
    model_py = args.model_py or os.path.join(os.path.dirname(os.path.abspath(__file__)), "Model", "model.py")
    if not os.path.isfile(model_py):
        raise FileNotFoundError(f"model_py not found: {model_py}")

    module = import_model_module(model_py)
    ModelCls = pick_model_class(module, args.model_class)
    LOG.info(f"Using model class: {ModelCls.__name__}")

    # instantiate (try common ctor patterns)
    model = None
    ctor_trials = [
        {"in_channels": X_train_sig.shape[1], "num_classes": num_classes, "feat_dim": (X_train_feat.shape[1] if use_features else 0)},
        {"num_classes": num_classes},
        {"n_classes": num_classes},
        {"classes": num_classes},
        {},
    ]
    last_err = None
    for kw in ctor_trials:
        try:
            model = ModelCls(**kw)
            break
        except Exception as e:
            last_err = e
            model = None
    if model is None:
        raise RuntimeError(f"Model instantiation failed. Last error: {last_err}")

    model = model.to(device)
    LOG.info(f"Trainable params: {count_params(model):,}")

    # optimizer / scheduler / early stop
    criterion = LabelSmoothingCE(args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args)
    early = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    best_val_acc = -1.0
    best_path = os.path.join(run_dir, "best_model.pth")
    last_path = os.path.join(run_dir, "last_model.pth")

    # resume (optional)
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        if "optim" in ckpt:
            optimizer.load_state_dict(ckpt["optim"])
        if "sched" in ckpt and scheduler is not None:
            try:
                scheduler.load_state_dict(ckpt["sched"])
            except Exception:
                pass
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_acc = float(ckpt.get("best_val_acc", -1.0))
        LOG.info(f"Resumed from {args.resume} | start_epoch={start_epoch} | best_val_acc={best_val_acc:.4f}")

    # training loop
    history: List[Dict[str, Any]] = []
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        tr = train_one_epoch(
            model=model, loader=dl_train, optimizer=optimizer, device=device,
            criterion=criterion, grad_clip=args.grad_clip,
            use_amp=bool(args.amp and device.type == "cuda"),
            use_features=use_features,
            use_learned_filters=args.use_learned_filters,
            filt0=filt0, filt1=filt1, filt2=filt2
        )
        va = evaluate_loss_acc(
            model=model, loader=dl_val, device=device,
            criterion=criterion,
            use_features=use_features,
            use_learned_filters=args.use_learned_filters,
            filt0=filt0, filt1=filt1, filt2=filt2
        )

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(va["acc"])
            else:
                scheduler.step()

        lr_now = float(optimizer.param_groups[0]["lr"])
        dt = time.time() - t0

        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": tr["loss"],
            "train_acc": tr["acc"],
            "val_loss": va["loss"],
            "val_acc": va["acc"],
            "time_sec": dt,
            "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        }
        history.append(row)

        LOG.info(
            f"Epoch {epoch:03d}/{args.epochs} | lr={lr_now:.3e} | "
            f"tr_loss={tr['loss']:.4f} tr_acc={tr['acc']:.4f} | "
            f"va_loss={va['loss']:.4f} va_acc={va['acc']:.4f} | "
            f"{dt:.1f}s | cudnn={torch.backends.cudnn.enabled}"
        )

        # save checkpoints
        ckpt = {
            "epoch": epoch,
            "seed": seed,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "args": vars(args),
        }
        if scheduler is not None:
            ckpt["sched"] = scheduler.state_dict()
        torch.save(ckpt, last_path)

        if va["acc"] > best_val_acc:
            best_val_acc = va["acc"]
            torch.save(ckpt, best_path)
            LOG.info(f"New best model saved. best_val_acc={best_val_acc:.4f}")

        # persist history each epoch
        save_json({"history": history}, os.path.join(run_dir, "history.json"))
        # also csv-like
        try:
            import csv
            with open(os.path.join(run_dir, "history.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
                w.writeheader()
                w.writerows(history)
        except Exception:
            pass

        if early.step(va["acc"]):
            LOG.warning(f"Early stopping triggered at epoch {epoch}. Best val acc={best_val_acc:.4f}")
            break

    # plots (training curves)
    plot_training_curves(history, run_dir)

    # load best and evaluate on test
    LOG.info("Loading best checkpoint for final evaluation...")
    best = torch.load(best_path, map_location="cpu")
    model.load_state_dict(best["model"], strict=False)
    model.to(device)

    te = evaluate_loss_acc(
        model=model, loader=dl_test, device=device,
        criterion=criterion,
        use_features=use_features,
        use_learned_filters=args.use_learned_filters,
        filt0=filt0, filt1=filt1, filt2=filt2
    )
    LOG.info(f"[TEST] loss={te['loss']:.4f} acc={te['acc']:.4f}")

    # test predictions + paper figures
    logits, ytrue = predict_logits(
        model, dl_test, device,
        use_features=use_features,
        use_learned_filters=args.use_learned_filters,
        filt0=filt0, filt1=filt1, filt2=filt2
    )
    ypred = np.argmax(logits, axis=1)

    cm = confusion_matrix(ytrue, ypred, labels=list(range(num_classes)))
    plot_confusion_matrix(cm, classes_list, "Confusion Matrix (TEST)", os.path.join(run_dir, "cm_test.png"))
    per_class = plot_per_class_metrics(ytrue, ypred, classes_list, os.path.join(run_dir, "per_class_metrics_test.png"))

    # classification report (text)
    rep_txt = classification_report(ytrue, ypred, target_names=classes_list, zero_division=0)
    with open(os.path.join(run_dir, "classification_report_test.txt"), "w", encoding="utf-8") as f:
        f.write(rep_txt)

    # SNR evaluation (if present)
    snr_report: Dict[str, Dict[str, float]] = {}
    snr_acc_curve: Dict[str, float] = {}

    snr_keys = [k for k in data.keys() if k.startswith("test_snr_") and k.endswith("_sig")]
    if len(snr_keys) > 0:
        LOG.info("Found SNR test sets in NPZ. Evaluating...")
        for ks in sorted(snr_keys):
            snr = ks.replace("test_snr_", "").replace("_sig", "")
            kf = ks.replace("_sig", "_feat")

            Xs = data[ks].astype(np.float32)
            ys = y_test

            if Xs.shape[0] != ys.shape[0]:
                LOG.warning(f"SNR {snr}: shape mismatch X={Xs.shape[0]} vs y={ys.shape[0]}, skipping")
                continue

            if use_features and (kf in data):
                Xf = data[kf].astype(np.float32)
                Xf = apply_feature_scaler(Xf, scaler_mean, scaler_scale)
                ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(Xf), torch.from_numpy(ys))
                use_feat_here = True
            else:
                ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys))
                use_feat_here = False

            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin, drop_last=False)

            rep = evaluate_loss_acc(
                model=model, loader=dl, device=device,
                criterion=criterion,
                use_features=use_feat_here,
                use_learned_filters=args.use_learned_filters,
                filt0=filt0, filt1=filt1, filt2=filt2
            )
            snr_report[snr] = rep
            snr_acc_curve[snr] = rep["acc"]
            LOG.info(f"[SNR {snr}] loss={rep['loss']:.4f} acc={rep['acc']:.4f}")

            # SNR confusion matrix
            slogits, sytrue = predict_logits(
                model, dl, device,
                use_features=use_feat_here,
                use_learned_filters=args.use_learned_filters,
                filt0=filt0, filt1=filt1, filt2=filt2
            )
            sypred = np.argmax(slogits, axis=1)
            scm = confusion_matrix(sytrue, sypred, labels=list(range(num_classes)))
            plot_confusion_matrix(scm, classes_list, f"Confusion Matrix (SNR {snr})", os.path.join(run_dir, f"cm_snr_{snr}.png"))

        plot_snr_curve(snr_acc_curve, os.path.join(run_dir, "snr_curve.png"))

    # final report
    final_report = {
        "seed": seed,
        "num_classes": num_classes,
        "classes": classes_list,
        "test": te,
        "snr_report": snr_report,
        "per_class_metrics_test": per_class,
        "best_val_acc": float(best.get("best_val_acc", -1.0)),
        "run_dir": run_dir,
    }
    save_json(final_report, os.path.join(run_dir, "final_report.json"))

    return {
        "seed": seed,
        "run_dir": run_dir,
        "test_acc": float(te["acc"]),
        "test_loss": float(te["loss"]),
        "snr_report": snr_report,
    }


# -----------------------------
# Main (multi-seed)
# -----------------------------
def main():
    # VSCode friendliness: inject defaults if no CLI args
    if len(sys.argv) == 1:
        default_npz = r"D:\Bearing Project\4. Bearings\IMS\processed_ims\processed_ims_full.npz"
        default_out = r"D:\Bearing Project\4. Bearings\IMS\runs_ims"
        sys.argv += [
            "--npz", default_npz,
            "--out_dir", default_out,
            "--seeds", "42", "101", "2024",
            "--epochs", "60",
            "--batch_size", "64",
            "--lr", "1e-3",
            "--scheduler", "cosine",
            "--use_learned_filters",
            "--filter_ch0", r"D:\Bearing Project\4. Bearings\IMS\processed_ims\learned_filter_ims_ch0.pt",
            "--filter_ch1", r"D:\Bearing Project\4. Bearings\IMS\processed_ims\learned_filter_ims_ch1.pt",
            "--filter_ch2", r"D:\Bearing Project\4. Bearings\IMS\processed_ims\learned_filter_ims_ch2.pt",
        ]

    args = build_argparser().parse_args()

    seeds = args.seeds if (args.seeds is not None and len(args.seeds) > 0) else [args.seed]

    LOG.info("==============================================")
    LOG.info(f"NPZ: {args.npz}")
    LOG.info(f"OUT_DIR: {args.out_dir}")
    LOG.info(f"Seeds: {seeds}")
    LOG.info("==============================================")

    all_runs: List[Dict[str, Any]] = []

    for s in seeds:
        LOG.info(f"\n==================== RUN seed={s} ====================")
        res = run_one_seed(args, s)
        all_runs.append(res)
        LOG.info(f"[SEED {s}] TEST acc={res['test_acc']:.4f} loss={res['test_loss']:.4f}")

    # summary mean ± std over seeds
    accs = np.array([r["test_acc"] for r in all_runs], dtype=np.float64)
    mean_acc = float(accs.mean())
    std_acc = float(accs.std(ddof=1)) if len(accs) > 1 else 0.0

    LOG.info("\n==================== SUMMARY ====================")
    LOG.info(f"TEST Accuracy = {mean_acc:.4f} ± {std_acc:.4f} (mean ± std over {len(seeds)} runs)")

    # aggregate SNR mean±std
    all_snrs = sorted({snr for r in all_runs for snr in r["snr_report"].keys()})
    snr_summary: Dict[str, Dict[str, Any]] = {}
    for snr in all_snrs:
        vals = []
        for r in all_runs:
            if snr in r["snr_report"]:
                vals.append(r["snr_report"][snr]["acc"])
        if len(vals) > 0:
            v = np.array(vals, dtype=np.float64)
            snr_summary[snr] = {
                "mean_acc": float(v.mean()),
                "std_acc": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "n": int(len(v)),
            }
            LOG.info(f"SNR {snr}: acc = {snr_summary[snr]['mean_acc']:.4f} ± {snr_summary[snr]['std_acc']:.4f} (n={len(v)})")

    # save global summary
    summary_dir = ensure_dir(os.path.join(args.out_dir, "summary"))
    summary = {
        "npz": args.npz,
        "seeds": seeds,
        "test_acc_mean": mean_acc,
        "test_acc_std": std_acc,
        "runs": [{"seed": r["seed"], "run_dir": r["run_dir"], "test_acc": r["test_acc"], "test_loss": r["test_loss"]} for r in all_runs],
        "snr_summary": snr_summary,
        "timestamp": now_stamp(),
    }
    summary_path = os.path.join(summary_dir, f"summary_{now_stamp()}.json")
    save_json(summary, summary_path)

    # plot global mean±std bar + snr curve (mean±std)
    plot_seed_summary_bar(mean_acc, std_acc, seeds, os.path.join(summary_dir, "summary_test_mean_std.png"))

    if len(snr_summary) > 0:
        # plot mean curve only (std in text/json; keeping plot clean for paper)
        snr_mean_curve = {k: snr_summary[k]["mean_acc"] for k in snr_summary.keys()}
        plot_snr_curve(snr_mean_curve, os.path.join(summary_dir, "summary_snr_curve_mean.png"))

    LOG.info(f"\nGlobal summary saved: {summary_path}")
    LOG.info("All done")


if __name__ == "__main__":
    main()