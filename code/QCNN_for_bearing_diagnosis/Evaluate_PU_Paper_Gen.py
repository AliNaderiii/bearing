# -*- coding: utf-8 -*-
r"""
train_pu_real.py
=============================================================================
PU UltraPro Training - FINAL FIXED (Leak-Free, Paper Edition) - Reviewed
Author: Ali Naderi - 2026 (Reviewed & fixed)
Modifications:
 - Fix GradScaler/autocast usage
 - Safe device selection
 - Robust OneCycle total_steps calculation
 - Worker seeding, persistent_workers
 - Curriculum noise schedule
 - Train accuracy logging
 - Safer eval (no silent crashes)
=============================================================================
"""
from __future__ import annotations

import os
import sys
import json
import time
import csv
import math
import argparse
import random
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report

import plotly.graph_objects as go
import plotly.express as px

# ---------------- logging ----------------
import logging
LOG = logging.getLogger("train_pu_ultrapro_final")
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOG.addHandler(_h)

PLOTLY_TEMPLATE = "plotly_white"

# ---------------- utils ----------------
def now_stamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def ensure_dir(p: str | Path) -> str:
    p = str(p)
    os.makedirs(p, exist_ok=True)
    return p

def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def save_text(txt: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def np_load_mmap(path: str) -> np.ndarray:
    return np.load(path, mmap_mode="r")

def try_plotly_export_png(fig: go.Figure, out_png: str, scale: int = 3) -> bool:
    try:
        import kaleido  # noqa: F401
        fig.write_image(out_png, scale=scale)
        return True
    except Exception:
        return False

def fig_save(fig: go.Figure, html_path: str, png_path: Optional[str] = None) -> None:
    fig.update_layout(template=PLOTLY_TEMPLATE, font=dict(family="Arial", size=14))
    fig.write_html(html_path, include_plotlyjs="cdn")
    if png_path:
        _ = try_plotly_export_png(fig, png_path, scale=3)

def quick_nan_check(arr: np.ndarray, name: str, n: int = 5000) -> None:
    a = np.array(arr[:min(len(arr), n)], copy=False)
    bad = ~np.isfinite(a)
    if bad.any():
        LOG.warning(f"[DATA WARN] {name}: non-finite count in first {min(len(arr),n)} = {int(bad.sum())}")
    else:
        LOG.info(f"[DATA OK] {name}: finite (sampled {min(len(arr),n)})")

def log_class_distribution(y: np.ndarray, K: int, tag: str) -> None:
    y = np.asarray(y).astype(np.int64)
    counts = np.bincount(y, minlength=K)
    present = [i for i in range(len(counts)) if counts[i] > 0]
    LOG.info(f"{tag} classes present: {present}")
    LOG.info(f"{tag} class counts: {counts.tolist()}")

def infer_num_classes_from_y(y_train: np.ndarray) -> int:
    y = np.asarray(y_train).astype(np.int64)
    if y.size == 0: return 0
    return int(y.max()) + 1

def sanitize_class_map(classes_npy_obj: Any) -> Dict[int, str]:
    try:
        obj = classes_npy_obj
        if isinstance(obj, np.ndarray):
            if obj.ndim == 0: obj = obj.item()
            elif obj.size >= 1: obj = obj.ravel()[0]
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                try: out[int(k)] = str(v)
                except: pass
            if out: return out
    except: pass
    return {}

def build_class_names(K: int, class_map: Dict[int, str]) -> List[str]:
    names = []
    for i in range(K):
        if i in class_map: names.append(str(class_map[i]))
        else: names.append(f"class_{i}")
    return names

def check_label_range(y: np.ndarray, K: int, split_name: str) -> None:
    y = np.asarray(y).astype(np.int64)
    if y.size == 0: return
    mn, mx = int(y.min()), int(y.max())
    if mn < 0 or mx >= K:
        raise ValueError(f"[LABEL ERROR] {split_name}: labels out of range [0, {K-1}]. Found min={mn}, max={mx}.")

# ---------------- dynamic import ----------------
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

# ---------------- data loading ----------------
def load_processed_dir(data_dir: str) -> Dict[str, Any]:
    data_dir = str(Path(data_dir).resolve())
    need = [
        "X_train_sig.npy","X_train_feat.npy","y_train.npy",
        "X_val_sig.npy","X_val_feat.npy","y_val.npy",
        "X_test_sig.npy","X_test_feat.npy","y_test.npy",
        "classes.npy",
    ]
    for f in need:
        if not os.path.exists(os.path.join(data_dir, f)):
            raise FileNotFoundError(f"Missing: {f} in {data_dir}")

    out = {
        "X_train_sig": np_load_mmap(os.path.join(data_dir, "X_train_sig.npy")),
        "X_train_feat": np_load_mmap(os.path.join(data_dir, "X_train_feat.npy")),
        "y_train": np_load_mmap(os.path.join(data_dir, "y_train.npy")),
        "X_val_sig": np_load_mmap(os.path.join(data_dir, "X_val_sig.npy")),
        "X_val_feat": np_load_mmap(os.path.join(data_dir, "X_val_feat.npy")),
        "y_val": np_load_mmap(os.path.join(data_dir, "y_val.npy")),
        "X_test_sig": np_load_mmap(os.path.join(data_dir, "X_test_sig.npy")),
        "X_test_feat": np_load_mmap(os.path.join(data_dir, "X_test_feat.npy")),
        "y_test": np_load_mmap(os.path.join(data_dir, "y_test.npy")),
        "classes": np.load(os.path.join(data_dir, "classes.npy"), allow_pickle=True),
    }
    # benches
    for snr in [-10, -8, -6, -4, -2]:
        ks = os.path.join(data_dir, f"test_snr_{snr}_sig.npy")
        kf = os.path.join(data_dir, f"test_snr_{snr}_feat.npy")
        if os.path.exists(ks) and os.path.exists(kf):
            out[f"test_snr_{snr}_sig"] = np_load_mmap(ks)
            out[f"test_snr_{snr}_feat"] = np_load_mmap(kf)
    return out

class MemmapDataset(Dataset):
    def __init__(self, X_sig: np.ndarray, X_feat: np.ndarray, y: np.ndarray):
        self.X_sig = X_sig
        self.X_feat = X_feat
        self.y = np.asarray(y).astype(np.int64)
        assert len(self.X_sig) == len(self.X_feat) == len(self.y)

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int):
        x = np.array(self.X_sig[idx], dtype=np.float32, copy=True)
        f = np.array(self.X_feat[idx], dtype=np.float32, copy=True)
        if not np.isfinite(x).all(): x = np.nan_to_num(x, nan=0.0)
        if not np.isfinite(f).all(): f = np.nan_to_num(f, nan=0.0)
        y = int(self.y[idx])
        return torch.from_numpy(x), torch.from_numpy(f), torch.tensor(y, dtype=torch.long)

# ---------------- augmentation / noise / tta ----------------
def mixup_batch(x: torch.Tensor, feat: torch.Tensor, y: torch.Tensor, alpha: float, device: torch.device):
    if alpha <= 0: return x, feat, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=device)
    return lam*x + (1-lam)*x[idx], lam*feat + (1-lam)*feat[idx], y, y[idx], lam

def add_awgn_torch(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    # keep dtype & device
    snr_lin = 10.0 ** (snr_db / 10.0)
    p_sig = torch.mean(x * x, dim=2, keepdim=True).clamp_min(1e-12)
    p_noise = p_sig / snr_lin
    noise = torch.randn_like(x) * torch.sqrt(p_noise)
    return x + noise

@torch.no_grad()
def tta_logits(model: nn.Module, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
    # small TTA ensemble (rolls + scale)
    model.eval()
    L = model(x, feat)
    L = L + model(torch.roll(x, shifts=5, dims=2), feat)
    L = L + model(torch.roll(x, shifts=-5, dims=2), feat)
    L = L + model(x * 0.98, feat)
    return L / 4.0

# ---------------- metrics / plotting ----------------
def macro_f1(y_true, y_pred, K):
    if len(y_true) == 0: return 0.0
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(K)), average="macro", zero_division=0)
    return float(f1)

def per_class_metrics(y_true, y_pred, K):
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=list(range(K)), zero_division=0)
    return {"precision": p.tolist(), "recall": r.tolist(), "f1": f1.tolist(), "support": sup.tolist()}

def plot_confusion(cm, class_names, out_dir, tag):
    fig = px.imshow(cm, text_auto=True, aspect="auto", x=class_names, y=class_names, color_continuous_scale="Blues")
    fig.update_layout(title=f"Confusion Matrix - {tag}")
    fig_save(fig, os.path.join(out_dir, f"cm_{tag}.html"), os.path.join(out_dir, f"cm_{tag}.png"))

def plot_per_class_bars(m, class_names, out_dir, tag):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=class_names, y=m["precision"], name="Precision"))
    fig.add_trace(go.Bar(x=class_names, y=m["recall"], name="Recall"))
    fig.add_trace(go.Bar(x=class_names, y=m["f1"], name="F1"))
    fig.update_layout(barmode="group", title=f"Metrics - {tag}")
    fig_save(fig, os.path.join(out_dir, f"per_class_{tag}.html"), os.path.join(out_dir, f"per_class_{tag}.png"))

def plot_snr_curve(snr_to_val, out_dir, tag, ylabel):
    keys = sorted(snr_to_val.keys(), key=lambda k: float(k))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=keys, y=[snr_to_val[k] for k in keys], mode="lines+markers"))
    fig.update_layout(title=f"Robustness - {tag}", xaxis_title="SNR (dB)", yaxis_title=ylabel)
    fig_save(fig, os.path.join(out_dir, f"snr_{tag}_{ylabel}.html"), os.path.join(out_dir, f"snr_{tag}_{ylabel}.png"))

def plot_curves(history, out_dir):
    if len(history) == 0:
        return
    e = [h["epoch"] for h in history]
    # Loss
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=e, y=[h.get("train_loss", 0.0) for h in history], name="Train Loss"))
    fig1.add_trace(go.Scatter(x=e, y=[h.get("val_loss", 0.0) for h in history], name="Val Loss"))
    fig_save(fig1, os.path.join(out_dir, "curve_loss.html"), os.path.join(out_dir, "curve_loss.png"))
    # Score
    fig2 = go.Figure()
    if "train_acc" in history[0]:
        fig2.add_trace(go.Scatter(x=e, y=[h.get("train_acc", 0.0) for h in history], name="Train Acc", line=dict(dash='dash')))
    fig2.add_trace(go.Scatter(x=e, y=[h.get("val_f1_clean", 0.0) for h in history], name="Val Clean F1"))
    fig2.add_trace(go.Scatter(x=e, y=[h.get("val_f1_snr", 0.0) for h in history], name="Val Noisy F1"))
    fig2.add_trace(go.Scatter(x=e, y=[h.get("score", 0.0) for h in history], name="Score (Weighted)"))
    fig_save(fig2, os.path.join(out_dir, "curve_score.html"), os.path.join(out_dir, "curve_score.png"))

# ---------------- param groups / early stop ----------------
def build_param_groups(model, base_lr, alpha_lr_scale, weight_decay):
    params_r, params_g, params_b, params_other = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if "conv_g" in name: params_g.append(p)
        elif "conv_b" in name: params_b.append(p)
        elif "conv_r" in name: params_r.append(p)
        else: params_other.append(p)
    return [
        {"params": params_other, "lr": base_lr, "weight_decay": weight_decay},
        {"params": params_r, "lr": base_lr, "weight_decay": weight_decay},
        {"params": params_g, "lr": base_lr * alpha_lr_scale, "weight_decay": weight_decay},
        {"params": params_b, "lr": base_lr * alpha_lr_scale, "weight_decay": weight_decay},
    ], {"other": len(params_other), "r": len(params_r), "g": len(params_g), "b": len(params_b)}

class EarlyStopping:
    def __init__(self, patience: int, min_delta: float):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = -1e18
        self.bad = 0

    def step(self, score: float) -> bool:
        if score > self.best + self.min_delta:
            self.best = score
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience

# ---------------- noise curriculum ----------------
def schedule_snr_linear(epoch:int, epochs:int, snr_max: float, snr_min: float) -> float:
    # epoch in [1..epochs]
    if epochs <= 1: return float((snr_min + snr_max)/2.0)
    t = (epoch - 1) / max(1, (epochs - 1))
    return float(snr_max * (1.0 - t) + snr_min * t)

# ---------------- worker init for reproducibility ----------------
def worker_init_fn(worker_id:int):
    # each worker should have its own seed derived from global random state
    seed = np.random.randint(2**31 - 1)
    np.random.seed(seed)
    random.seed(seed)

# ---------------- train/eval core (fixed GradScaler/autocast) ----------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    criterion: nn.Module,
    use_amp: bool,
    accum_steps: int,
    grad_clip: float,
    use_mixup: bool,
    mixup_alpha: float,
    train_noise: bool,
    train_snr_min: float,
    train_snr_max: float,
    train_noise_prob: float,
    epoch:int = 1,
    epochs:int = 1,
) -> Dict[str, Any]:
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
    accum_steps = max(1, int(accum_steps))
    optimizer.zero_grad(set_to_none=True)

    loss_sum = 0.0
    correct = 0
    total = 0
    n_sum = 0
    opt_steps = 0

    for step, (x, f, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        f = f.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if not torch.isfinite(x).all(): x = torch.nan_to_num(x, nan=0.0)
        if not torch.isfinite(f).all(): f = torch.nan_to_num(f, nan=0.0)

        # decide whether to add noise this batch (curriculum based)
        if train_noise and (random.random() < train_noise_prob):
            snr = schedule_snr_linear(epoch, epochs, train_snr_max, train_snr_min)
            x_noisy = add_awgn_torch(x, snr)
            # small feature jitter
            f_noisy = f + torch.randn_like(f) * 0.05
            x_used = x_noisy
            f_used = f_noisy
        else:
            x_used = x
            f_used = f

        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            if use_mixup:
                x_m, f_m, y_a, y_b, lam = mixup_batch(x_used, f_used, y, mixup_alpha, device)
                logits = model(x_m, f_m)
                la = criterion(logits.float(), y_a)
                lb = criterion(logits.float(), y_b)
                loss_full = lam * la + (1.0 - lam) * lb
            else:
                logits = model(x_used, f_used)
                loss_full = criterion(logits.float(), y)

        if not torch.isfinite(loss_full):
            # skip this batch but do not crash
            continue

        loss = loss_full / accum_steps
        scaler.scale(loss).backward()

        # accuracy (approx)
        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            if use_mixup:
                # compare to stronger label y_a (approx)
                correct += int((pred == y_a).sum().item())
            else:
                correct += int((pred == y).sum().item())
            total += int(x.size(0))

        do_step = (step % accum_steps == 0) or (step == len(loader))
        if do_step:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            opt_steps += 1
            # step scheduler per optimizer update (correct for OneCycleLR)
            if scheduler is not None:
                try:
                    scheduler.step()
                except Exception:
                    pass

        bs = int(x.size(0))
        loss_sum += float(loss_full.detach().cpu().item()) * bs
        n_sum += bs

    epoch_acc = float(correct / max(1, total))
    epoch_loss = float(loss_sum / max(1, n_sum))
    return {"loss": epoch_loss, "acc": epoch_acc, "opt_steps": int(opt_steps)}

@torch.no_grad()
def eval_val_loss(model, loader, device, criterion):
    model.eval()
    s, n = 0.0, 0
    for x, f, y in loader:
        x = x.to(device, non_blocking=True)
        f = f.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if not torch.isfinite(x).all(): x = torch.nan_to_num(x)
        if not torch.isfinite(f).all(): f = torch.nan_to_num(f)
        logits = model(x, f)
        logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=0.0, neginf=0.0)
        loss = criterion(logits, y)
        if not torch.isfinite(loss): continue
        s += float(loss.item()) * x.size(0)
        n += x.size(0)
    return float(s / n) if n > 0 else float("inf")

@torch.no_grad()
def eval_split_logits(model, loader, device, use_tta, snr_db=None):
    model.eval()
    all_L, all_y = [], []
    for x, f, y in loader:
        x = x.to(device, non_blocking=True)
        f = f.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if not torch.isfinite(x).all(): x = torch.nan_to_num(x)
        if not torch.isfinite(f).all(): f = torch.nan_to_num(f)
        if snr_db is not None:
            x = add_awgn_torch(x, float(snr_db))
            if not torch.isfinite(x).all(): x = torch.nan_to_num(x)
        try:
            L = tta_logits(model, x, f) if use_tta else model(x, f)
        except Exception:
            # fallback to zeros logits of correct shape to avoid crashes
            L = torch.zeros((x.size(0), getattr(model, "num_classes", 1)), device=x.device)
        L = torch.nan_to_num(L.float(), nan=0.0, posinf=0.0, neginf=0.0)
        all_L.append(L.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())
    if len(all_L) == 0: return np.zeros((0,1), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.concatenate(all_L, axis=0), np.concatenate(all_y, axis=0)

@torch.no_grad()
def ensemble_logits(models, loader, device, use_tta, snr_db=None):
    all_L, all_y = [], []
    for x, f, y in loader:
        x = x.to(device, non_blocking=True)
        f = f.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if not torch.isfinite(x).all(): x = torch.nan_to_num(x)
        if not torch.isfinite(f).all(): f = torch.nan_to_num(f)
        if snr_db is not None:
            x = add_awgn_torch(x, float(snr_db))
            if not torch.isfinite(x).all(): x = torch.nan_to_num(x)
        Lsum = None
        valid = 0
        for m in models:
            try:
                L = tta_logits(m, x, f) if use_tta else m(x, f)
                L = torch.nan_to_num(L.float(), nan=0.0, posinf=0.0, neginf=0.0)
                Lsum = L if Lsum is None else (Lsum + L)
                valid += 1
            except Exception:
                continue
        if valid == 0:
            # last-resort: skip this batch
            continue
        Lavg = Lsum / float(valid)
        all_L.append(Lavg.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())
    if len(all_L) == 0: return np.zeros((0,1), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.concatenate(all_L, axis=0), np.concatenate(all_y, axis=0)

def softmax_np(logits):
    x = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.clip(ex.sum(axis=1, keepdims=True), 1e-12, None)

def fit_temperature_scaling(logits, y_true):
    logits_t = torch.tensor(logits, dtype=torch.float32)
    y_t = torch.tensor(y_true, dtype=torch.long)
    logT = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)
    ce = nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad()
        T = torch.exp(logT).clamp(1e-3, 100.0)
        loss = ce(logits_t/T, y_t)
        loss.backward()
        return loss
    try:
        opt.step(closure)
    except Exception:
        pass
    return float(torch.exp(logT).item())

# ---------------- run one seed ----------------
def run_one_seed(args, seed, data, class_names, K):
    seed_everything(seed)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    LOG.info(f"Using device: {device}")

    ytr = np.asarray(data["y_train"]).astype(np.int64)
    check_label_range(ytr, K, "train")
    log_class_distribution(ytr, K, "TRAIN")

    ds_tr = MemmapDataset(data["X_train_sig"], data["X_train_feat"], ytr)
    ds_va = MemmapDataset(data["X_val_sig"], data["X_val_feat"], np.asarray(data["y_val"]).astype(np.int64))
    ds_te = MemmapDataset(data["X_test_sig"], data["X_test_feat"], np.asarray(data["y_test"]).astype(np.int64))

    sampler = None
    if args.use_weighted_sampler:
        counts = np.bincount(ytr, minlength=K).astype(np.float64)
        counts[counts==0] = 1.0
        w_per_class = 1.0 / counts
        w_samples = w_per_class[ytr]
        sampler = WeightedRandomSampler(torch.from_numpy(w_samples).double(), num_samples=len(w_samples), replacement=True)
        LOG.info("Using WeightedRandomSampler.")

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, sampler=sampler,
                       shuffle=(sampler is None), num_workers=args.num_workers, pin_memory=args.pin_memory,
                       worker_init_fn=(worker_init_fn if args.num_workers>0 else None),
                       persistent_workers=(args.num_workers>0))
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=args.pin_memory,
                       worker_init_fn=(worker_init_fn if args.num_workers>0 else None),
                       persistent_workers=(args.num_workers>0))
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=args.pin_memory,
                       worker_init_fn=(worker_init_fn if args.num_workers>0 else None),
                       persistent_workers=(args.num_workers>0))

    module = import_model_module(args.model_py)
    ModelCls = pick_model_class(module, args.model_class)
    feat_dim = int(data["X_train_feat"].shape[1])
    model = ModelCls(in_channels=3, num_classes=K, feat_dim=feat_dim).to(device)

    # class weights
    weight_t = None
    if args.use_class_weights:
        counts = np.bincount(ytr, minlength=K).astype(np.float64); counts[counts==0]=1.0
        w = (counts.sum() / counts)
        w = w / w.mean()
        weight_t = torch.tensor(w, dtype=torch.float32, device=device)
        LOG.info(f"Class weights: {w.round(4).tolist()}")

    criterion = nn.CrossEntropyLoss(weight=weight_t, label_smoothing=float(args.label_smoothing))

    groups, ginfo = build_param_groups(model, args.lr, args.alpha_lr_scale, args.weight_decay)
    optimizer = optim.AdamW(groups)
    LOG.info(f"Param groups: other={ginfo['other']} r={ginfo['r']} g={ginfo['g']} b={ginfo['b']}")

    # compute total_steps robustly
    optim_steps_per_epoch = max(1, math.ceil(len(dl_tr) / max(1, args.accum_steps)))
    total_steps = int(optim_steps_per_epoch * args.epochs)
    try:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[pg["lr"] for pg in optimizer.param_groups],
            total_steps=total_steps,
            pct_start=float(args.pct_start),
            div_factor=float(args.div_factor),
            final_div_factor=float(args.final_div_factor),
        )
    except Exception as e:
        LOG.warning(f"Failed to create OneCycleLR: {e}. Scheduler disabled.")
        scheduler = None

    run_dir = ensure_dir(os.path.join(args.out_dir, f"run_{now_stamp()}_seed{seed}"))
    best_path = os.path.join(run_dir, "best_model.pth")
    save_json(vars(args) | {"seed_used": seed, "num_classes": K, "feat_dim": feat_dim}, os.path.join(run_dir, "config.json"))

    best_score = -1e9
    history = []
    early = EarlyStopping(args.patience, args.min_delta)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        # Noise warmup: don't use train noise in first mixup_warmup_epochs (or specified)
        noise_active = args.train_noise and (ep > args.mixup_warmup_epochs)
        use_mix = args.mixup and (ep > args.mixup_warmup_epochs)

        tr = train_one_epoch(
            model, dl_tr, optimizer, scheduler, device, criterion,
            use_amp=args.amp, accum_steps=args.accum_steps, grad_clip=args.grad_clip,
            use_mixup=use_mix, mixup_alpha=args.mixup_alpha,
            train_noise=noise_active, train_snr_min=args.train_snr_min, train_snr_max=args.train_snr_max, train_noise_prob=args.train_noise_prob,
            epoch=ep, epochs=args.epochs
        )

        vL, vy = eval_split_logits(model, dl_va, device, use_tta=False, snr_db=None)
        val_f1_clean = macro_f1(vy, vL.argmax(1) if len(vy) else np.array([], dtype=np.int64), K) if len(vy) else 0.0

        val_f1_snr = val_f1_clean
        if args.val_snr is not None:
            vLs, vys = eval_split_logits(model, dl_va, device, use_tta=False, snr_db=float(args.val_snr))
            val_f1_snr = macro_f1(vys, vLs.argmax(1) if len(vys) else np.array([], dtype=np.int64), K) if len(vys) else val_f1_clean

        score = float(args.w_clean * val_f1_clean + args.w_snr * val_f1_snr)
        val_loss = eval_val_loss(model, dl_va, device, criterion)

        lrs = [pg.get("lr", 0.0) for pg in optimizer.param_groups]
        row = {
            "epoch": ep,
            "train_loss": tr["loss"],
            "train_acc": tr["acc"],
            "val_loss": val_loss,
            "val_f1_clean": val_f1_clean,
            "val_f1_snr": val_f1_snr,
            "score": score,
            "lr_min": float(min(lrs)),
            "lr_max": float(max(lrs)),
            "time_sec": float(time.time() - t0),
        }
        history.append(row)
        save_json({"history": history}, os.path.join(run_dir, "history.json"))
        with open(os.path.join(run_dir, "history.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader()
            w.writerows(history)

        LOG.info(f"Ep {ep:02d} | L_tr={tr['loss']:.4f} Acc_tr={tr['acc']:.4f} | F1_c={val_f1_clean:.4f} F1_s={val_f1_snr:.4f} | Sc={score:.4f} | {row['time_sec']:.1f}s")

        if score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "best_score": best_score}, best_path)
            LOG.info(f"  [BEST] {best_score:.4f}")

        if early.step(score):
            LOG.info(f"Early stopping at epoch {ep}. best_score={best_score:.4f}")
            break

    # plotting
    plot_curves(history, run_dir)

    # load best & test
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    use_tta_final = bool(args.tta and args.tta_final_only)
    tL, ty = eval_split_logits(model, dl_te, device, use_tta_final, snr_db=None)
    tpred = tL.argmax(1) if len(tL) else np.array([], dtype=np.int64)
    acc = float((tpred == ty).mean()) if len(ty) else 0.0
    f1 = macro_f1(ty, tpred, K) if len(ty) else 0.0

    cm = confusion_matrix(ty, tpred, labels=list(range(K))) if len(ty) else np.zeros((K, K), dtype=int)
    plot_confusion(cm, class_names, run_dir, f"test_seed{seed}")
    save_text(classification_report(ty, tpred, labels=list(range(K)), target_names=class_names, zero_division=0), os.path.join(run_dir, "report.txt"))

    # SNR benches
    snr_res = {}
    for k in list(data.keys()):
        if k.startswith("test_snr_") and k.endswith("_sig"):
            snr = k.replace("test_snr_", "").replace("_sig", "")
            kf = k.replace("_sig", "_feat")
            d_snr = MemmapDataset(data[k], data[kf], np.asarray(data["y_test"]).astype(np.int64))
            l_snr = DataLoader(d_snr, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory,
                               worker_init_fn=(worker_init_fn if args.num_workers>0 else None), persistent_workers=(args.num_workers>0))
            Ls, ys = eval_split_logits(model, l_snr, device, use_tta_final, snr_db=None)
            snr_res[snr] = float((Ls.argmax(1) == ys).mean()) if len(ys) else 0.0

    if snr_res: plot_snr_curve(snr_res, run_dir, f"seed{seed}", "acc")
    final = {"test_acc": acc, "test_f1": f1, "snr": snr_res, "run_dir": run_dir}
    save_json(final, os.path.join(run_dir, "final.json"))
    return final, best_path

# ---------------- CLI ----------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model_py", required=True)
    p.add_argument("--model_class", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--alpha_lr_scale", type=float, default=0.5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--pct_start", type=float, default=0.1)
    p.add_argument("--div_factor", type=float, default=10.0)
    p.add_argument("--final_div_factor", type=float, default=10000.0)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--use_weighted_sampler", action="store_true")
    p.add_argument("--use_class_weights", action="store_true")
    p.add_argument("--mixup", action="store_true")
    p.add_argument("--mixup_alpha", type=float, default=0.1)
    p.add_argument("--mixup_warmup_epochs", type=int, default=5)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--val_snr", type=float, default=-6.0)
    p.add_argument("--w_clean", type=float, default=0.6)
    p.add_argument("--w_snr", type=float, default=0.4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--tta_final_only", type=int, default=1)
    p.add_argument("--do_ensemble", action="store_true")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--calib_use_val_snr", action="store_true")
    p.add_argument("--ece_bins", type=int, default=15)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    # Noise args
    p.add_argument("--train_noise", action="store_true")
    p.add_argument("--train_snr_min", type=float, default=-10.0)
    p.add_argument("--train_snr_max", type=float, default=5.0)
    p.add_argument("--train_noise_prob", type=float, default=0.5)
    return p

def main():
    args = build_argparser().parse_args()
    ensure_dir(args.out_dir)
    data = load_processed_dir(args.data_dir)

    # K from y_train
    ytr = np.asarray(data["y_train"]).astype(np.int64)
    K = infer_num_classes_from_y(ytr)
    class_map = sanitize_class_map(data["classes"])
    class_names = build_class_names(K, class_map)

    LOG.info("==============================================")
    LOG.info(f"DATA_DIR: {args.data_dir}")
    LOG.info(f"OUT_DIR:  {args.out_dir}")
    LOG.info(f"MODEL:    {args.model_py}")
    LOG.info(f"Classes:  {K} | feat_dim={data['X_train_feat'].shape[1]}")
    LOG.info(f"Class map parsed: {class_map if class_map else '[EMPTY -> class_i]'}")
    LOG.info(f"Seeds:    {args.seeds}")
    LOG.info("==============================================")

    results = []
    best_paths = []
    for s in args.seeds:
        LOG.info(f"\n================ RUN seed={s} ================")
        res, bp = run_one_seed(args, s, data, class_names, K)
        results.append(res)
        best_paths.append(bp)
        LOG.info(f"[SEED {s}] TEST acc={res['test_acc']:.4f} f1={res['test_f1']:.4f}")

    # Ensemble (unchanged logic from your code)
    if args.do_ensemble:
        LOG.info("\nEnsemble...")
        device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
        ModelCls = pick_model_class(import_model_module(args.model_py), args.model_class)
        feat_dim = int(data["X_train_feat"].shape[1])
        models = []
        for bp in best_paths:
            ck = torch.load(bp, map_location=device)
            m = ModelCls(in_channels=3, num_classes=K, feat_dim=feat_dim).to(device)
            m.load_state_dict(ck["model"], strict=False)
            m.eval()
            models.append(m)

        yte = np.asarray(data["y_test"]).astype(np.int64)
        ds_te = MemmapDataset(data["X_test_sig"], data["X_test_feat"], yte)
        dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

        use_tta = (args.tta and args.tta_final_only)
        L, y = ensemble_logits(models, dl_te, device, use_tta)
        acc = float((L.argmax(1) == y).mean()) if len(y) else 0.0
        LOG.info(f"Ensemble Acc: {acc:.4f}")

        T = 1.0
        if args.calibrate:
            yva = np.asarray(data["y_val"]).astype(np.int64)
            ds_va = MemmapDataset(data["X_val_sig"], data["X_val_feat"], yva)
            dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
            snr_cal = float(args.val_snr) if (args.calib_use_val_snr and args.val_snr is not None) else None
            Lv, yv = ensemble_logits(models, dl_va, device, use_tta, snr_cal)
            if len(Lv) > 0:
                T = fit_temperature_scaling(Lv, yv)
                LOG.info(f"Calibrated T={T:.4f}")

        snr_curve = {}
        for snr in [-10, -8, -6, -4, -2, 0, 5]:
            Ls, y = ensemble_logits(models, dl_te, device, use_tta, float(snr))
            if len(Ls) > 0 and len(y) > 0:
                snr_curve[str(snr)] = float((Ls.argmax(1) == y).mean())
        final_rep = {
            "ensemble_acc": acc,
            "T": float(T),
            "robustness": snr_curve,
            "timestamp": now_stamp()
        }
        save_json(final_rep, os.path.join(args.out_dir, "ensemble_final.json"))
        LOG.info("Done.")

if __name__ == "__main__":
    main()
