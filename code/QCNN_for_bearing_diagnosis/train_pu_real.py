# # -*- coding: utf-8 -*-
# r"""
# train_pu_real.py
# =============================================================================
# PU UltraPro Training - FINAL (Leak-Free, Paper Edition) - Compact8 Ready
# Author: Ali Naderi - 2026

# Works with your preprocessing output:
#   - X_train_sig.npy, X_train_feat.npy, y_train.npy (memmap)
#   - X_val_sig.npy,   X_val_feat.npy,   y_val.npy
#   - X_test_sig.npy,  X_test_feat.npy,  y_test.npy
#   - classes.npy  (dict id->name, saved as np.array([dict], dtype=object))
#   - test_snr_-10_sig.npy, test_snr_-10_feat.npy, ... (optional benches)

# Key stability features:
# - K (num_classes) ALWAYS inferred from y_train (robust)
# - classes.npy parsed robustly (works even if malformed)
# - label range sanity check before training (prevents CUDA CE crash)
# - SAFE evaluation: never drops all batches (sanitize logits instead of skipping)
# - OneCycleLR total_steps correct + grad accumulation
# - MixUp warmup
# - Robust ensemble + calibration with fail-safe

# Outputs (per seed):
# - run_.../best_model.pth, last_model.pth
# - run_.../history.csv, history.json
# - run_.../final_report.json
# - run_.../cm_test_*.html/png, per_class_*.html/png
# - run_.../snr_curve_*.html/png (if benches exist)

# Outputs (ensemble):
# - ensemble_.../final_ensemble_report.json
# - ensemble_.../reliability_*.html/png
# - ensemble_.../cm_ensemble_*.html/png
# =============================================================================
# """

# from __future__ import annotations

# import os
# import sys
# import json
# import time
# import csv
# import math
# import argparse
# import random
# import datetime
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple

# import numpy as np

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report

# import plotly.graph_objects as go
# import plotly.express as px


# # =============================================================================
# # Logging
# # =============================================================================
# import logging
# LOG = logging.getLogger("train_pu_ultrapro_final")
# LOG.setLevel(logging.INFO)
# _h = logging.StreamHandler(sys.stdout)
# _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
# LOG.handlers.clear()
# LOG.addHandler(_h)

# PLOTLY_TEMPLATE = "plotly_white"


# # =============================================================================
# # General Utilities
# # =============================================================================
# def now_stamp() -> str:
#     return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# def ensure_dir(p: str | Path) -> str:
#     p = str(p)
#     os.makedirs(p, exist_ok=True)
#     return p


# def save_json(obj: Any, path: str) -> None:
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(obj, f, indent=2, ensure_ascii=False)


# def save_text(txt: str, path: str) -> None:
#     with open(path, "w", encoding="utf-8") as f:
#         f.write(txt)


# def seed_everything(seed: int) -> None:
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# def np_load_mmap(path: str) -> np.ndarray:
#     return np.load(path, mmap_mode="r")


# def safe_item(x: Any) -> Any:
#     if isinstance(x, np.ndarray) and x.shape == ():
#         return x.item()
#     return x


# def quick_nan_check(arr: np.ndarray, name: str, n: int = 5000) -> None:
#     a = np.array(arr[:min(len(arr), n)], copy=False)
#     bad = ~np.isfinite(a)
#     if bad.any():
#         LOG.warning(f"[DATA WARN] {name}: non-finite count in first {min(len(arr),n)} = {int(bad.sum())}")
#     else:
#         LOG.info(f"[DATA OK] {name}: finite (sampled {min(len(arr),n)})")


# def infer_num_classes_from_y(y_train: np.ndarray) -> int:
#     y = np.asarray(y_train).astype(np.int64)
#     if y.size == 0:
#         return 0
#     return int(y.max()) + 1


# def sanitize_class_map(classes_npy_obj: Any) -> Dict[int, str]:
#     """
#     Robust parsing for classes.npy saved as np.array([dict], dtype=object) or dict with str/int keys.
#     Returns dict[int,str]. If parsing fails, returns {}.
#     """
#     try:
#         obj = classes_npy_obj
#         if isinstance(obj, np.ndarray):
#             # typical: array([dict], dtype=object)
#             if obj.dtype == object and obj.size == 1:
#                 obj = obj.item()
#             elif obj.size == 1:
#                 obj = obj.item()
#         if isinstance(obj, dict):
#             out = {}
#             for k, v in obj.items():
#                 try:
#                     out[int(k)] = str(v)
#                 except Exception:
#                     continue
#             return out
#     except Exception:
#         pass
#     return {}


# def build_class_names(K: int, class_map: Dict[int, str]) -> List[str]:
#     return [class_map[i] if i in class_map else f"class_{i}" for i in range(K)]


# def check_label_range(y: np.ndarray, K: int, split_name: str) -> None:
#     y = np.asarray(y).astype(np.int64)
#     if y.size == 0:
#         return
#     mn, mx = int(y.min()), int(y.max())
#     if mn < 0 or mx >= K:
#         raise ValueError(
#             f"[LABEL ERROR] {split_name}: labels out of range for K={K}. "
#             f"min={mn}, max={mx}."
#         )


# def log_class_distribution(y: np.ndarray, K: int, tag: str) -> None:
#     y = np.asarray(y).astype(np.int64)
#     counts = np.bincount(y, minlength=K)
#     present = [i for i in range(K) if counts[i] > 0]
#     LOG.info(f"{tag} classes present: {present}")
#     LOG.info(f"{tag} class counts: {counts.tolist()}")


# # =============================================================================
# # Plotly helpers
# # =============================================================================
# def try_plotly_export_png(fig: go.Figure, out_png: str, scale: int = 3) -> bool:
#     try:
#         import kaleido  # noqa: F401
#         fig.write_image(out_png, scale=scale)
#         return True
#     except Exception:
#         return False


# def fig_save(fig: go.Figure, html_path: str, png_path: Optional[str] = None) -> None:
#     fig.update_layout(template=PLOTLY_TEMPLATE, font=dict(family="Arial", size=14))
#     fig.write_html(html_path, include_plotlyjs="cdn")
#     if png_path:
#         _ = try_plotly_export_png(fig, png_path, scale=3)


# def plot_confusion(cm: np.ndarray, class_names: List[str], out_dir: str, tag: str) -> None:
#     fig = px.imshow(cm, text_auto=True, aspect="auto",
#                     x=class_names, y=class_names, color_continuous_scale="Blues")
#     fig.update_layout(title=f"Confusion Matrix - {tag}", xaxis_title="Predicted", yaxis_title="True")
#     fig_save(fig, os.path.join(out_dir, f"cm_{tag}.html"), os.path.join(out_dir, f"cm_{tag}.png"))


# def plot_per_class_bars(metrics: Dict[str, Any], class_names: List[str], out_dir: str, tag: str) -> None:
#     fig = go.Figure()
#     fig.add_trace(go.Bar(x=class_names, y=metrics["precision"], name="Precision"))
#     fig.add_trace(go.Bar(x=class_names, y=metrics["recall"], name="Recall"))
#     fig.add_trace(go.Bar(x=class_names, y=metrics["f1"], name="F1"))
#     fig.update_layout(barmode="group", title=f"Per-class Metrics - {tag}", yaxis=dict(range=[0, 1.02]))
#     fig_save(fig, os.path.join(out_dir, f"per_class_{tag}.html"), os.path.join(out_dir, f"per_class_{tag}.png"))


# def plot_snr_curve(snr_to_val: Dict[str, float], out_dir: str, tag: str, ylabel: str) -> None:
#     keys = sorted(snr_to_val.keys(), key=lambda k: float(k))
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=keys, y=[snr_to_val[k] for k in keys], mode="lines+markers"))
#     fig.update_layout(title=f"SNR Robustness - {tag} ({ylabel})", xaxis_title="SNR (dB)", yaxis_title=ylabel, yaxis=dict(range=[0, 1.02]))
#     fig_save(fig, os.path.join(out_dir, f"snr_{tag}_{ylabel}.html"), os.path.join(out_dir, f"snr_{tag}_{ylabel}.png"))


# def plot_curves(history: List[Dict[str, Any]], out_dir: str) -> None:
#     e = [h["epoch"] for h in history]

#     fig1 = go.Figure()
#     fig1.add_trace(go.Scatter(x=e, y=[h["train_loss"] for h in history], mode="lines+markers", name="Train Loss"))
#     fig1.add_trace(go.Scatter(x=e, y=[h["val_loss"] for h in history], mode="lines+markers", name="Val Loss"))
#     fig1.update_layout(title="Loss Curves", xaxis_title="Epoch", yaxis_title="Loss")
#     fig_save(fig1, os.path.join(out_dir, "curve_loss.html"), os.path.join(out_dir, "curve_loss.png"))

#     fig2 = go.Figure()
#     fig2.add_trace(go.Scatter(x=e, y=[h["val_f1_clean"] for h in history], mode="lines+markers", name="Val Macro-F1 (clean)"))
#     fig2.add_trace(go.Scatter(x=e, y=[h["val_f1_snr"] for h in history], mode="lines+markers", name="Val Macro-F1 (snr)"))
#     fig2.add_trace(go.Scatter(x=e, y=[h["score"] for h in history], mode="lines+markers", name="Score"))
#     fig2.update_layout(title="Selection Metrics", xaxis_title="Epoch", yaxis_title="Score", yaxis=dict(range=[0, 1.02]))
#     fig_save(fig2, os.path.join(out_dir, "curve_score.html"), os.path.join(out_dir, "curve_score.png"))


# # =============================================================================
# # Dynamic model import
# # =============================================================================
# def import_model_module(model_py: str) -> Any:
#     import importlib.util
#     model_py = os.path.abspath(model_py)
#     spec = importlib.util.spec_from_file_location("user_model_module", model_py)
#     if spec is None or spec.loader is None:
#         raise RuntimeError(f"Could not load model module from: {model_py}")
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#     return module


# def pick_model_class(module: Any, preferred: Optional[str] = None) -> type:
#     if preferred and hasattr(module, preferred):
#         cls = getattr(module, preferred)
#         if isinstance(cls, type) and issubclass(cls, nn.Module):
#             return cls
#     for name in ["QDCA_Hybrid_Pro", "QDCA_Hybrid", "QDCA_Net", "Model", "Net"]:
#         if hasattr(module, name):
#             cls = getattr(module, name)
#             if isinstance(cls, type) and issubclass(cls, nn.Module):
#                 return cls
#     for _, obj in vars(module).items():
#         if isinstance(obj, type) and issubclass(obj, nn.Module):
#             return obj
#     raise RuntimeError("No nn.Module subclass found in model module.")


# # =============================================================================
# # Load processed dir
# # =============================================================================
# def load_processed_dir(data_dir: str) -> Dict[str, Any]:
#     data_dir = str(Path(data_dir).resolve())
#     need = [
#         "X_train_sig.npy","X_train_feat.npy","y_train.npy",
#         "X_val_sig.npy","X_val_feat.npy","y_val.npy",
#         "X_test_sig.npy","X_test_feat.npy","y_test.npy",
#         "classes.npy",
#     ]
#     for f in need:
#         p = os.path.join(data_dir, f)
#         if not os.path.exists(p):
#             raise FileNotFoundError(f"Missing: {p}")

#     out = {
#         "X_train_sig": np_load_mmap(os.path.join(data_dir, "X_train_sig.npy")),
#         "X_train_feat": np_load_mmap(os.path.join(data_dir, "X_train_feat.npy")),
#         "y_train": np_load_mmap(os.path.join(data_dir, "y_train.npy")),
#         "X_val_sig": np_load_mmap(os.path.join(data_dir, "X_val_sig.npy")),
#         "X_val_feat": np_load_mmap(os.path.join(data_dir, "X_val_feat.npy")),
#         "y_val": np_load_mmap(os.path.join(data_dir, "y_val.npy")),
#         "X_test_sig": np_load_mmap(os.path.join(data_dir, "X_test_sig.npy")),
#         "X_test_feat": np_load_mmap(os.path.join(data_dir, "X_test_feat.npy")),
#         "y_test": np_load_mmap(os.path.join(data_dir, "y_test.npy")),
#         "classes": np.load(os.path.join(data_dir, "classes.npy"), allow_pickle=True),
#     }

#     # benches
#     for snr in [-10, -8, -6, -4, -2]:
#         ks = os.path.join(data_dir, f"test_snr_{snr}_sig.npy")
#         kf = os.path.join(data_dir, f"test_snr_{snr}_feat.npy")
#         if os.path.exists(ks) and os.path.exists(kf):
#             out[f"test_snr_{snr}_sig"] = np_load_mmap(ks)
#             out[f"test_snr_{snr}_feat"] = np_load_mmap(kf)

#     return out


# # =============================================================================
# # Dataset
# # =============================================================================
# class MemmapDataset(Dataset):
#     def __init__(self, X_sig: np.ndarray, X_feat: np.ndarray, y: np.ndarray):
#         self.X_sig = X_sig
#         self.X_feat = X_feat
#         self.y = np.asarray(y).astype(np.int64)
#         assert len(self.X_sig) == len(self.X_feat) == len(self.y)

#     def __len__(self) -> int:
#         return int(len(self.y))

#     def __getitem__(self, idx: int):
#         x = np.array(self.X_sig[idx], dtype=np.float32, copy=True)
#         f = np.array(self.X_feat[idx], dtype=np.float32, copy=True)
#         if not np.isfinite(x).all():
#             x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
#         if not np.isfinite(f).all():
#             f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
#         y = int(self.y[idx])
#         return torch.from_numpy(x), torch.from_numpy(f), torch.tensor(y, dtype=torch.long)


# # =============================================================================
# # Aug / Noise / TTA
# # =============================================================================
# def mixup_batch(x: torch.Tensor, feat: torch.Tensor, y: torch.Tensor, alpha: float, device: torch.device):
#     if alpha <= 0:
#         return x, feat, y, y, 1.0
#     lam = np.random.beta(alpha, alpha)
#     idx = torch.randperm(x.size(0), device=device)
#     return lam*x + (1-lam)*x[idx], lam*feat + (1-lam)*feat[idx], y, y[idx], lam


# def add_awgn_torch(x: torch.Tensor, snr_db: float) -> torch.Tensor:
#     snr_lin = 10.0 ** (snr_db / 10.0)
#     p_sig = torch.mean(x * x, dim=2, keepdim=True).clamp_min(1e-12)
#     p_noise = p_sig / snr_lin
#     noise = torch.randn_like(x) * torch.sqrt(p_noise)
#     return x + noise


# @torch.no_grad()
# def tta_logits(model: nn.Module, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
#     L = model(x, feat)
#     L = L + model(torch.roll(x, shifts=5, dims=2), feat)
#     L = L + model(torch.roll(x, shifts=-5, dims=2), feat)
#     L = L + model(x * 0.98, feat)
#     return L / 4.0


# # =============================================================================
# # Metrics
# # =============================================================================
# def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, K: int) -> float:
#     _, _, f1, _ = precision_recall_fscore_support(
#         y_true, y_pred, labels=list(range(K)), average="macro", zero_division=0
#     )
#     return float(f1)


# def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, K: int) -> Dict[str, Any]:
#     p, r, f1, sup = precision_recall_fscore_support(
#         y_true, y_pred, labels=list(range(K)), zero_division=0
#     )
#     return {"precision": p.tolist(), "recall": r.tolist(), "f1": f1.tolist(), "support": sup.tolist()}


# # =============================================================================
# # Param grouping (CWRU style)
# # =============================================================================
# def build_param_groups(model: nn.Module, base_lr: float, alpha_lr_scale: float, weight_decay: float):
#     params_r, params_g, params_b, params_other = [], [], [], []
#     for name, p in model.named_parameters():
#         if not p.requires_grad:
#             continue
#         if "conv_g" in name:
#             params_g.append(p)
#         elif "conv_b" in name:
#             params_b.append(p)
#         elif "conv_r" in name:
#             params_r.append(p)
#         else:
#             params_other.append(p)

#     groups = [
#         {"params": params_other, "lr": base_lr, "weight_decay": weight_decay},
#         {"params": params_r, "lr": base_lr, "weight_decay": weight_decay},
#         {"params": params_g, "lr": base_lr * alpha_lr_scale, "weight_decay": weight_decay},
#         {"params": params_b, "lr": base_lr * alpha_lr_scale, "weight_decay": weight_decay},
#     ]
#     info = {"other": len(params_other), "r": len(params_r), "g": len(params_g), "b": len(params_b)}
#     return groups, info


# # =============================================================================
# # Early stopping
# # =============================================================================
# class EarlyStopping:
#     def __init__(self, patience: int, min_delta: float):
#         self.patience = int(patience)
#         self.min_delta = float(min_delta)
#         self.best = -1e18
#         self.bad = 0

#     def step(self, score: float) -> bool:
#         if score > self.best + self.min_delta:
#             self.best = score
#             self.bad = 0
#             return False
#         self.bad += 1
#         return self.bad >= self.patience


# # =============================================================================
# # Train/Eval core (SAFE EVAL)
# # =============================================================================
# def train_one_epoch(model, loader, optimizer, scheduler, device, criterion,
#                     use_amp: bool, accum_steps: int, grad_clip: float,
#                     use_mixup: bool, mixup_alpha: float) -> Dict[str, Any]:
#     model.train()
#     scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
#     accum_steps = max(1, int(accum_steps))
#     optimizer.zero_grad(set_to_none=True)

#     loss_sum, n_sum, opt_steps = 0.0, 0, 0

#     for step, (x, f, y) in enumerate(loader, start=1):
#         x = x.to(device, non_blocking=True)
#         f = f.to(device, non_blocking=True)
#         y = y.to(device, non_blocking=True)

#         with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
#             if use_mixup:
#                 x_m, f_m, y_a, y_b, lam = mixup_batch(x, f, y, mixup_alpha, device)
#                 logits = model(x_m, f_m)
#                 la = criterion(logits.float(), y_a)
#                 lb = criterion(logits.float(), y_b)
#                 loss_full = lam * la + (1.0 - lam) * lb
#             else:
#                 logits = model(x, f)
#                 loss_full = criterion(logits.float(), y)

#         if not torch.isfinite(loss_full):
#             continue

#         bs = int(x.size(0))
#         loss_sum += float(loss_full.detach().cpu().item()) * bs
#         n_sum += bs

#         loss = loss_full / accum_steps
#         scaler.scale(loss).backward()

#         do_step = (step % accum_steps == 0) or (step == len(loader))
#         if do_step:
#             if grad_clip and grad_clip > 0:
#                 scaler.unscale_(optimizer)
#                 nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

#             scaler.step(optimizer)
#             scaler.update()
#             optimizer.zero_grad(set_to_none=True)
#             opt_steps += 1
#             if scheduler is not None:
#                 scheduler.step()

#     return {"loss": float(loss_sum / max(1, n_sum)), "opt_steps": int(opt_steps)}


# @torch.no_grad()
# def eval_val_loss(model, loader, device, criterion) -> float:
#     """
#     SAFE eval loss: sanitize logits instead of skipping batches.
#     Never returns inf due to all-batch skip (returns large finite value if n==0).
#     """
#     model.eval()
#     s, n = 0.0, 0
#     for x, f, y in loader:
#         x = x.to(device, non_blocking=True)
#         f = f.to(device, non_blocking=True)
#         y = y.to(device, non_blocking=True)

#         logits = model(x, f)
#         logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=0.0, neginf=0.0)

#         loss = criterion(logits, y)
#         if not torch.isfinite(loss):
#             continue

#         bs = int(x.size(0))
#         s += float(loss.item()) * bs
#         n += bs

#     return float(s / n) if n > 0 else 1e9


# @torch.no_grad()
# def eval_split_logits(model, loader, device, use_tta: bool, snr_db: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     SAFE eval logits: always returns full-length arrays (no skipping).
#     """
#     model.eval()
#     all_L, all_y = [], []
#     for x, f, y in loader:
#         x = x.to(device, non_blocking=True)
#         f = f.to(device, non_blocking=True)
#         y = y.to(device, non_blocking=True)

#         if snr_db is not None:
#             x = add_awgn_torch(x, float(snr_db))
#             x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

#         L = tta_logits(model, x, f) if use_tta else model(x, f)
#         L = torch.nan_to_num(L.float(), nan=0.0, posinf=0.0, neginf=0.0)

#         all_L.append(L.detach().cpu().numpy())
#         all_y.append(y.detach().cpu().numpy())

#     return np.concatenate(all_L, axis=0), np.concatenate(all_y, axis=0)


# # =============================================================================
# # Ensemble + Calibration (fail-safe)
# # =============================================================================
# @torch.no_grad()
# def ensemble_logits(models: List[nn.Module],
#                     loader: DataLoader,
#                     device: torch.device,
#                     use_tta: bool,
#                     snr_db: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
#     all_L, all_y = [], []
#     fallback_batches = 0
#     skipped_batches = 0

#     for x, f, y in loader:
#         x = x.to(device, non_blocking=True)
#         f = f.to(device, non_blocking=True)
#         y = y.to(device, non_blocking=True)

#         if snr_db is not None:
#             x = add_awgn_torch(x, float(snr_db))
#             x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

#         Lsum = None
#         valid = 0
#         for m in models:
#             try:
#                 L = tta_logits(m, x, f) if use_tta else m(x, f)
#             except Exception:
#                 continue
#             L = torch.nan_to_num(L.float(), nan=0.0, posinf=0.0, neginf=0.0)
#             Lsum = L if Lsum is None else (Lsum + L)
#             valid += 1

#         if valid == 0:
#             fallback_batches += 1
#             try:
#                 L = models[0](x, f).float()
#                 L = torch.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)
#                 Lsum = L
#                 valid = 1
#             except Exception:
#                 skipped_batches += 1
#                 continue

#         Lavg = Lsum / float(valid)
#         all_L.append(Lavg.detach().cpu().numpy())
#         all_y.append(y.detach().cpu().numpy())

#     if len(all_L) == 0:
#         LOG.warning("[ENSEMBLE] Empty logits (all batches failed).")
#         return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64)

#     if fallback_batches > 0 or skipped_batches > 0:
#         LOG.warning(f"[ENSEMBLE] fallback_batches={fallback_batches}, skipped_batches={skipped_batches}")

#     return np.concatenate(all_L, axis=0), np.concatenate(all_y, axis=0)


# def softmax_np(logits: np.ndarray) -> np.ndarray:
#     x = logits - logits.max(axis=1, keepdims=True)
#     ex = np.exp(x)
#     return ex / np.clip(ex.sum(axis=1, keepdims=True), 1e-12, None)


# def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> Tuple[float, Dict[str, Any]]:
#     conf = probs.max(axis=1)
#     pred = probs.argmax(axis=1)
#     acc = (pred == y_true).astype(np.float32)
#     bins = np.linspace(0.0, 1.0, n_bins + 1)

#     ece = 0.0
#     stats = []
#     for i in range(n_bins):
#         lo, hi = bins[i], bins[i + 1]
#         m = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
#         if m.sum() == 0:
#             stats.append({"lo": float(lo), "hi": float(hi), "count": 0, "acc": None, "conf": None})
#             continue
#         ba = float(acc[m].mean())
#         bc = float(conf[m].mean())
#         w = float(m.mean())
#         ece += abs(ba - bc) * w
#         stats.append({"lo": float(lo), "hi": float(hi), "count": int(m.sum()), "acc": ba, "conf": bc})

#     return float(ece), {"bins": stats, "n_bins": n_bins}


# def plot_reliability(bin_info: Dict[str, Any], out_dir: str, tag: str) -> None:
#     xs, ys = [], []
#     for b in bin_info["bins"]:
#         if b["count"] and b["acc"] is not None and b["conf"] is not None:
#             xs.append(b["conf"]); ys.append(b["acc"])
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(dash="dash")))
#     fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+lines", name="Model"))
#     fig.update_layout(title=f"Reliability - {tag}", xaxis_title="Confidence", yaxis_title="Accuracy",
#                       xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
#     fig_save(fig, os.path.join(out_dir, f"reliability_{tag}.html"), os.path.join(out_dir, f"reliability_{tag}.png"))


# def fit_temperature_scaling(logits: np.ndarray, y_true: np.ndarray, max_iter: int = 200) -> float:
#     logits_t = torch.tensor(logits, dtype=torch.float32, device="cpu")
#     y_t = torch.tensor(y_true, dtype=torch.long, device="cpu")
#     logT = torch.tensor([0.0], dtype=torch.float32, requires_grad=True, device="cpu")
#     opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
#     ce = nn.CrossEntropyLoss()

#     def closure():
#         opt.zero_grad(set_to_none=True)
#         T = torch.exp(logT).clamp(1e-3, 100.0)
#         loss = ce(logits_t / T, y_t)
#         loss.backward()
#         return loss

#     opt.step(closure)
#     return float(torch.exp(logT).detach().cpu().item())


# # =============================================================================
# # One seed run
# # =============================================================================
# def run_one_seed(args: argparse.Namespace, seed: int, data: Dict[str, Any],
#                  ModelCls: type, class_names: List[str], K: int) -> Tuple[Dict[str, Any], str]:
#     seed_everything(seed)

#     device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
#     if device.type == "cuda":
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True

#     ytr = np.asarray(data["y_train"]).astype(np.int64)
#     yva = np.asarray(data["y_val"]).astype(np.int64)
#     yte = np.asarray(data["y_test"]).astype(np.int64)

#     check_label_range(ytr, K, "train")
#     check_label_range(yva, K, "val")
#     check_label_range(yte, K, "test")

#     log_class_distribution(ytr, K, "TRAIN")
#     log_class_distribution(yva, K, "VAL")
#     log_class_distribution(yte, K, "TEST")

#     feat_dim = int(data["X_train_feat"].shape[1])

#     ds_tr = MemmapDataset(data["X_train_sig"], data["X_train_feat"], ytr)
#     ds_va = MemmapDataset(data["X_val_sig"], data["X_val_feat"], yva)
#     ds_te = MemmapDataset(data["X_test_sig"], data["X_test_feat"], yte)

#     pin = bool(args.pin_memory and device.type == "cuda")

#     sampler = None
#     if args.use_weighted_sampler:
#         counts = np.bincount(ytr, minlength=K).astype(np.float64)
#         counts[counts == 0] = 1.0
#         w_per_class = 1.0 / counts
#         w_samples = w_per_class[ytr]
#         sampler = WeightedRandomSampler(torch.from_numpy(w_samples).double(), num_samples=len(w_samples), replacement=True)
#         LOG.info("Using WeightedRandomSampler.")

#     dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
#                        num_workers=args.num_workers, pin_memory=pin, drop_last=False,
#                        persistent_workers=(args.num_workers > 0))
#     dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
#                        num_workers=args.num_workers, pin_memory=pin, drop_last=False,
#                        persistent_workers=(args.num_workers > 0))
#     dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
#                        num_workers=args.num_workers, pin_memory=pin, drop_last=False,
#                        persistent_workers=(args.num_workers > 0))

#     model = ModelCls(in_channels=3, num_classes=K, feat_dim=feat_dim).to(device)

#     weight_t = None
#     if args.use_class_weights:
#         counts = np.bincount(ytr, minlength=K).astype(np.float64)
#         counts[counts == 0] = 1.0
#         w = (counts.sum() / counts)
#         w = w / w.mean()
#         weight_t = torch.tensor(w, dtype=torch.float32, device=device)
#         LOG.info(f"Class weights: {w.round(4).tolist()}")

#     criterion = nn.CrossEntropyLoss(weight=weight_t, label_smoothing=float(args.label_smoothing))

#     groups, ginfo = build_param_groups(model, args.lr, args.alpha_lr_scale, args.weight_decay)
#     optimizer = optim.AdamW(groups)
#     LOG.info(f"Param groups: other={ginfo['other']} r={ginfo['r']} g={ginfo['g']} b={ginfo['b']}")

#     accum_steps = max(1, int(args.accum_steps))
#     optim_steps_per_epoch = max(1, math.ceil(len(dl_tr) / accum_steps))
#     total_steps = int(optim_steps_per_epoch * args.epochs)

#     scheduler = optim.lr_scheduler.OneCycleLR(
#         optimizer,
#         max_lr=[pg["lr"] for pg in optimizer.param_groups],
#         total_steps=total_steps,
#         pct_start=float(args.pct_start),
#         div_factor=float(args.div_factor),
#         final_div_factor=float(args.final_div_factor),
#     )

#     run_dir = ensure_dir(os.path.join(args.out_dir, f"run_{now_stamp()}_seed{seed}"))
#     best_path = os.path.join(run_dir, "best_model.pth")
#     last_path = os.path.join(run_dir, "last_model.pth")
#     save_json(vars(args) | {"seed_used": seed, "num_classes": K, "feat_dim": feat_dim}, os.path.join(run_dir, "config.json"))

#     early = EarlyStopping(args.patience, args.min_delta)
#     history: List[Dict[str, Any]] = []
#     best_score = -1e18

#     for epoch in range(1, args.epochs + 1):
#         t0 = time.time()
#         use_mixup_now = bool(args.mixup and (epoch > args.mixup_warmup_epochs))

#         tr = train_one_epoch(
#             model, dl_tr, optimizer, scheduler, device, criterion,
#             use_amp=args.amp, accum_steps=args.accum_steps, grad_clip=args.grad_clip,
#             use_mixup=use_mixup_now, mixup_alpha=args.mixup_alpha
#         )

#         # val clean
#         vL, vy = eval_split_logits(model, dl_va, device, use_tta=False, snr_db=None)
#         vpred = np.argmax(vL, axis=1)
#         val_f1_clean = macro_f1(vy, vpred, K)

#         # val snr (val-only AWGN, no leakage)
#         if args.val_snr is not None:
#             vLs, vys = eval_split_logits(model, dl_va, device, use_tta=False, snr_db=float(args.val_snr))
#             vpreds = np.argmax(vLs, axis=1)
#             val_f1_snr = macro_f1(vys, vpreds, K)
#         else:
#             val_f1_snr = val_f1_clean

#         score = float(args.w_clean * val_f1_clean + args.w_snr * val_f1_snr)
#         val_loss = eval_val_loss(model, dl_va, device, criterion)

#         lr_list = [pg["lr"] for pg in optimizer.param_groups]
#         lr_min, lr_max = float(min(lr_list)), float(max(lr_list))
#         dt = time.time() - t0

#         row = {
#             "epoch": epoch,
#             "train_loss": tr["loss"],
#             "val_loss": val_loss,
#             "val_f1_clean": val_f1_clean,
#             "val_f1_snr": val_f1_snr,
#             "score": score,
#             "lr_min": lr_min,
#             "lr_max": lr_max,
#             "opt_steps": tr["opt_steps"],
#             "mixup_active": bool(use_mixup_now),
#             "time_sec": dt,
#         }
#         history.append(row)

#         LOG.info(
#             f"Ep {epoch:03d}/{args.epochs} | tr_loss={tr['loss']:.4f} val_loss={val_loss:.4f} | "
#             f"valF1m={val_f1_clean:.4f} valF1m_snr({args.val_snr})={val_f1_snr:.4f} | "
#             f"score={score:.4f} | lr=[{lr_min:.2e},{lr_max:.2e}] | steps={tr['opt_steps']} | {dt:.1f}s"
#         )

#         ckpt = {
#             "epoch": epoch,
#             "seed": seed,
#             "model": model.state_dict(),
#             "optim": optimizer.state_dict(),
#             "sched": scheduler.state_dict(),
#             "best_score": best_score,
#             "args": vars(args),
#         }
#         torch.save(ckpt, last_path)

#         if score > best_score:
#             best_score = score
#             ckpt["best_score"] = best_score
#             torch.save(ckpt, best_path)
#             LOG.info(f"[BEST] score={best_score:.4f}")

#         # persist history
#         save_json({"history": history}, os.path.join(run_dir, "history.json"))
#         with open(os.path.join(run_dir, "history.csv"), "w", newline="", encoding="utf-8") as f:
#             w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
#             w.writeheader()
#             w.writerows(history)

#         if early.step(best_score):
#             LOG.warning(f"Early stopping at epoch {epoch}. best_score={best_score:.4f}")
#             break

#     # curves
#     plot_curves(history, run_dir)

#     # final test (best)
#     best = torch.load(best_path, map_location="cpu")
#     model.load_state_dict(best["model"], strict=False)
#     model.to(device)
#     model.eval()

#     use_tta_final = bool(args.tta and args.tta_final_only)
#     tL, ty = eval_split_logits(model, dl_te, device, use_tta=use_tta_final, snr_db=None)
#     tpred = np.argmax(tL, axis=1)
#     test_acc = float((tpred == ty).mean())
#     test_f1m = macro_f1(ty, tpred, K)

#     cm = confusion_matrix(ty, tpred, labels=list(range(K)))
#     plot_confusion(cm, class_names, run_dir, f"test_seed{seed}")
#     plot_per_class_bars(per_class_metrics(ty, tpred, K), class_names, run_dir, f"test_seed{seed}")

#     rep_txt = classification_report(ty, tpred, labels=list(range(K)), target_names=class_names, zero_division=0)
#     save_text(rep_txt, os.path.join(run_dir, "classification_report_test.txt"))

#     # snr benches (from files, not runtime noise)
#     snr_acc, snr_f1 = {}, {}
#     for k in list(data.keys()):
#         if k.startswith("test_snr_") and k.endswith("_sig"):
#             snr = k.replace("test_snr_", "").replace("_sig", "")
#             kf = k.replace("_sig", "_feat")
#             if kf not in data:
#                 continue
#             ds = MemmapDataset(data[k], data[kf], yte)
#             dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin, drop_last=False)
#             Ls, ys = eval_split_logits(model, dl, device, use_tta=use_tta_final, snr_db=None)
#             yp = np.argmax(Ls, axis=1)
#             snr_acc[snr] = float((yp == ys).mean())
#             snr_f1[snr] = macro_f1(ys, yp, K)

#     if snr_acc:
#         plot_snr_curve(snr_acc, run_dir, f"seed{seed}", "acc")
#     if snr_f1:
#         plot_snr_curve(snr_f1, run_dir, f"seed{seed}", "macroF1")

#     final = {
#         "seed": seed,
#         "run_dir": run_dir,
#         "best_score": float(best_score),
#         "test_acc": test_acc,
#         "test_macroF1": test_f1m,
#         "tta_final": bool(use_tta_final),
#         "snr_acc": snr_acc,
#         "snr_macroF1": snr_f1,
#     }
#     save_json(final, os.path.join(run_dir, "final_report.json"))
#     return final, best_path


# # =============================================================================
# # CLI
# # =============================================================================
# def build_argparser() -> argparse.ArgumentParser:
#     p = argparse.ArgumentParser(description="PU UltraPro Training - FINAL (compact8-ready)")

#     p.add_argument("--data_dir", type=str, required=True)
#     p.add_argument("--out_dir", type=str, required=True)
#     p.add_argument("--model_py", type=str, required=True)
#     p.add_argument("--model_class", type=str, default=None)

#     p.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 2024])

#     p.add_argument("--epochs", type=int, default=40)
#     p.add_argument("--batch_size", type=int, default=64)
#     p.add_argument("--accum_steps", type=int, default=1)

#     p.add_argument("--lr", type=float, default=1e-3)
#     p.add_argument("--alpha_lr_scale", type=float, default=0.5)
#     p.add_argument("--weight_decay", type=float, default=1e-4)
#     p.add_argument("--grad_clip", type=float, default=1.0)

#     p.add_argument("--pct_start", type=float, default=0.10)
#     p.add_argument("--div_factor", type=float, default=10.0)
#     p.add_argument("--final_div_factor", type=float, default=10000.0)

#     p.add_argument("--label_smoothing", type=float, default=0.1)
#     p.add_argument("--use_weighted_sampler", action="store_true")
#     p.add_argument("--use_class_weights", action="store_true")

#     p.add_argument("--mixup", action="store_true")
#     p.add_argument("--mixup_alpha", type=float, default=0.1)
#     p.add_argument("--mixup_warmup_epochs", type=int, default=5)

#     p.add_argument("--amp", action="store_true")
#     p.add_argument("--pin_memory", action="store_true")
#     p.add_argument("--num_workers", type=int, default=0)
#     p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

#     p.add_argument("--val_snr", type=float, default=-6.0)
#     p.add_argument("--w_clean", type=float, default=0.6)
#     p.add_argument("--w_snr", type=float, default=0.4)

#     p.add_argument("--patience", type=int, default=12)
#     p.add_argument("--min_delta", type=float, default=0.0)

#     p.add_argument("--tta", action="store_true")
#     p.add_argument("--tta_final_only", type=int, default=1)

#     # ensemble + calibration
#     p.add_argument("--do_ensemble", action="store_true")
#     p.add_argument("--calibrate", action="store_true")
#     p.add_argument("--calib_use_val_snr", action="store_true")
#     p.add_argument("--ece_bins", type=int, default=15)

#     return p


# # =============================================================================
# # Main
# # =============================================================================
# def main():
#     args = build_argparser().parse_args()
#     ensure_dir(args.out_dir)

#     device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
#     if device.type == "cuda":
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True

#     data = load_processed_dir(args.data_dir)

#     quick_nan_check(data["X_val_sig"], "X_val_sig")
#     quick_nan_check(data["X_val_feat"], "X_val_feat")
#     quick_nan_check(data["X_test_sig"], "X_test_sig")
#     quick_nan_check(data["X_test_feat"], "X_test_feat")

#     # K is ALWAYS from y_train
#     ytr = np.asarray(data["y_train"]).astype(np.int64)
#     K = infer_num_classes_from_y(ytr)

#     class_map = sanitize_class_map(data["classes"])
#     class_names = build_class_names(K, class_map)

#     LOG.info("==============================================")
#     LOG.info(f"DATA_DIR: {args.data_dir}")
#     LOG.info(f"OUT_DIR:  {args.out_dir}")
#     LOG.info(f"MODEL:    {args.model_py}")
#     LOG.info(f"Classes:  {K} | feat_dim={data['X_train_feat'].shape[1]}")
#     LOG.info(f"Class map parsed: {class_map if class_map else '[EMPTY -> class_i]'}")
#     LOG.info(f"Seeds:    {args.seeds}")
#     LOG.info("==============================================")

#     module = import_model_module(args.model_py)
#     ModelCls = pick_model_class(module, args.model_class)

#     all_runs: List[Dict[str, Any]] = []
#     best_paths: List[str] = []

#     for s in args.seeds:
#         LOG.info(f"\n================ RUN seed={s} ================")
#         res, bp = run_one_seed(args, s, data, ModelCls, class_names, K)
#         all_runs.append(res)
#         best_paths.append(bp)
#         LOG.info(f"[SEED {s}] TEST acc={res['test_acc']:.4f} macroF1={res['test_macroF1']:.4f}")

#     summary_dir = ensure_dir(os.path.join(args.out_dir, "summary"))
#     save_json({"runs": all_runs, "best_paths": best_paths, "timestamp": now_stamp()}, os.path.join(summary_dir, f"summary_{now_stamp()}.json"))

#     # ---------------- ENSEMBLE (optional) ----------------
#     if not args.do_ensemble:
#         LOG.info("All done (no ensemble).")
#         return

#     LOG.info("\n================ STARTING ENSEMBLE ================")
#     feat_dim = int(data["X_train_feat"].shape[1])

#     models: List[nn.Module] = []
#     for bp in best_paths:
#         ckpt = torch.load(bp, map_location="cpu")
#         m = ModelCls(in_channels=3, num_classes=K, feat_dim=feat_dim).to(device)
#         m.load_state_dict(ckpt["model"], strict=False)
#         m.eval()
#         models.append(m)

#     yva = np.asarray(data["y_val"]).astype(np.int64)
#     yte = np.asarray(data["y_test"]).astype(np.int64)

#     pin = bool(args.pin_memory and device.type == "cuda")

#     ds_va = MemmapDataset(data["X_val_sig"], data["X_val_feat"], yva)
#     dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

#     ds_te = MemmapDataset(data["X_test_sig"], data["X_test_feat"], yte)
#     dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

#     use_tta_final = bool(args.tta and args.tta_final_only)
#     ens_dir = ensure_dir(os.path.join(args.out_dir, f"ensemble_{now_stamp()}"))

#     # uncalibrated
#     L_test, y_test_np = ensemble_logits(models, dl_te, device, use_tta=use_tta_final, snr_db=None)
#     if L_test.shape[0] == 0:
#         LOG.error("Ensemble logits empty on test. Aborting ensemble.")
#         return

#     P_uncal = softmax_np(L_test)
#     yp_uncal = P_uncal.argmax(axis=1)
#     acc_uncal = float((yp_uncal == y_test_np).mean())
#     f1_uncal = macro_f1(y_test_np, yp_uncal, K)

#     cm_u = confusion_matrix(y_test_np, yp_uncal, labels=list(range(K)))
#     plot_confusion(cm_u, class_names, ens_dir, "ensemble_test_uncal")
#     plot_per_class_bars(per_class_metrics(y_test_np, yp_uncal, K), class_names, ens_dir, "ensemble_test_uncal")

#     ece_uncal, bins_uncal = expected_calibration_error(P_uncal, y_test_np, n_bins=args.ece_bins)
#     plot_reliability(bins_uncal, ens_dir, "ensemble_test_uncal")

#     # calibration (val-only)
#     T = 1.0
#     if args.calibrate:
#         L_val, y_val_np = ensemble_logits(models, dl_va, device, use_tta=use_tta_final, snr_db=None)
#         if L_val.shape[0] == 0:
#             LOG.warning("[CALIB] val logits empty -> skip calibration.")
#             args.calibrate = False

#         if args.calibrate and args.calib_use_val_snr and (args.val_snr is not None):
#             L_val_s, y_val_s = ensemble_logits(models, dl_va, device, use_tta=use_tta_final, snr_db=float(args.val_snr))
#             if L_val_s.shape[0] == 0:
#                 LOG.warning("[CALIB] val_snr logits empty -> using clean val only.")
#                 args.calib_use_val_snr = False
#             else:
#                 L_fit = np.concatenate([L_val, L_val_s], axis=0)
#                 y_fit = np.concatenate([y_val_np, y_val_s], axis=0)
#                 T = fit_temperature_scaling(L_fit, y_fit, max_iter=200)
#                 save_json({"temperature": float(T)}, os.path.join(ens_dir, "temperature.json"))

#         if args.calibrate and (not args.calib_use_val_snr):
#             T = fit_temperature_scaling(L_val, y_val_np, max_iter=200)
#             save_json({"temperature": float(T)}, os.path.join(ens_dir, "temperature.json"))

#     # calibrated test
#     P_cal = softmax_np(L_test / float(T))
#     yp_cal = P_cal.argmax(axis=1)
#     acc_cal = float((yp_cal == y_test_np).mean())
#     f1_cal = macro_f1(y_test_np, yp_cal, K)

#     cm_c = confusion_matrix(y_test_np, yp_cal, labels=list(range(K)))
#     plot_confusion(cm_c, class_names, ens_dir, "ensemble_test_cal")
#     plot_per_class_bars(per_class_metrics(y_test_np, yp_cal, K), class_names, ens_dir, "ensemble_test_cal")

#     ece_cal, bins_cal = expected_calibration_error(P_cal, y_test_np, n_bins=args.ece_bins)
#     plot_reliability(bins_cal, ens_dir, "ensemble_test_cal")

#     # robustness from benches
#     snr_report = {}
#     for k in list(data.keys()):
#         if k.startswith("test_snr_") and k.endswith("_sig"):
#             snr = k.replace("test_snr_", "").replace("_sig", "")
#             kf = k.replace("_sig", "_feat")
#             if kf not in data:
#                 continue
#             ds = MemmapDataset(data[k], data[kf], yte)
#             dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

#             Ls, ys = ensemble_logits(models, dl, device, use_tta=use_tta_final, snr_db=None)
#             if Ls.shape[0] == 0:
#                 continue

#             Pu = softmax_np(Ls)
#             Pc = softmax_np(Ls / float(T))
#             ypu = Pu.argmax(axis=1)
#             ypc = Pc.argmax(axis=1)

#             snr_report[snr] = {
#                 "uncal": {"acc": float((ypu == ys).mean()), "macroF1": macro_f1(ys, ypu, K)},
#                 "cal":   {"acc": float((ypc == ys).mean()), "macroF1": macro_f1(ys, ypc, K)},
#             }

#     save_json({
#         "class_map": {str(i): class_names[i] for i in range(K)},
#         "ensemble": {
#             "tta_final": bool(use_tta_final),
#             "temperature": float(T),
#             "test_uncal": {"acc": acc_uncal, "macroF1": f1_uncal, "ece": ece_uncal},
#             "test_cal": {"acc": acc_cal, "macroF1": f1_cal, "ece": ece_cal},
#         },
#         "snr_report": snr_report,
#         "timestamp": now_stamp(),
#     }, os.path.join(ens_dir, "final_ensemble_report.json"))

#     LOG.info(f"[ENSEMBLE UNCAL] acc={acc_uncal:.4f} macroF1={f1_uncal:.4f} ECE={ece_uncal:.4f}")
#     LOG.info(f"[ENSEMBLE CAL]   acc={acc_cal:.4f} macroF1={f1_cal:.4f} ECE={ece_cal:.4f}")
#     LOG.info(f"Ensemble reports saved to: {ens_dir}")


# if __name__ == "__main__":
#     main()































# -*- coding: utf-8 -*-
r"""
train_pu_real.py
=============================================================================
PU UltraPro Training - FINAL FIXED (Leak-Free, Paper Edition)
Author: Ali Naderi - 2026

Fixed & Improved:
1. Fixed KeyError 'train_loss' bug.
2. Added Train Accuracy calculation to monitoring.
3. Implemented Curriculum Noise: Noise is disabled during warmup epochs
   to allow the model to stabilize before facing hard noise.
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report

import plotly.graph_objects as go
import plotly.express as px

import logging
LOG = logging.getLogger("train_pu_ultrapro_final")
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOG.addHandler(_h)

PLOTLY_TEMPLATE = "plotly_white"


# =============================================================================
# Utils
# =============================================================================
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

# =============================================================================
# Dynamic Model Import
# =============================================================================
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

# =============================================================================
# Data Loading
# =============================================================================
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

# =============================================================================
# Augmentation
# =============================================================================
def mixup_batch(x: torch.Tensor, feat: torch.Tensor, y: torch.Tensor, alpha: float, device: torch.device):
    if alpha <= 0: return x, feat, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=device)
    return lam*x + (1-lam)*x[idx], lam*feat + (1-lam)*feat[idx], y, y[idx], lam

def add_awgn_torch(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    snr_lin = 10.0 ** (snr_db / 10.0)
    p_sig = torch.mean(x * x, dim=2, keepdim=True).clamp_min(1e-12)
    p_noise = p_sig / snr_lin
    noise = torch.randn_like(x) * torch.sqrt(p_noise)
    return x + noise

@torch.no_grad()
def tta_logits(model: nn.Module, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
    model.eval()
    L = model(x, feat)
    L += model(torch.roll(x, shifts=5, dims=2), feat)
    L += model(torch.roll(x, shifts=-5, dims=2), feat)
    L += model(x * 0.98, feat)
    return L / 4.0

# =============================================================================
# Metrics
# =============================================================================
def macro_f1(y_true, y_pred, K):
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(K)), average="macro", zero_division=0)
    return float(f1)

def per_class_metrics(y_true, y_pred, K):
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=list(range(K)), zero_division=0)
    return {"precision": p.tolist(), "recall": r.tolist(), "f1": f1.tolist(), "support": sup.tolist()}

# =============================================================================
# Plotting
# =============================================================================
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
    fig.update_layout(title=f"Robustness - {tag}", xaxis_title="SNR", yaxis_title=ylabel)
    fig_save(fig, os.path.join(out_dir, f"snr_{tag}_{ylabel}.html"), os.path.join(out_dir, f"snr_{tag}_{ylabel}.png"))

def plot_curves(history, out_dir):
    e = [h["epoch"] for h in history]
    
    # 1. Loss Curve
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=e, y=[h["train_loss"] for h in history], name="Train Loss"))
    fig1.add_trace(go.Scatter(x=e, y=[h["val_loss"] for h in history], name="Val Loss"))
    fig_save(fig1, os.path.join(out_dir, "curve_loss.html"), os.path.join(out_dir, "curve_loss.png"))
    
    # 2. Score Curve (Acc/F1)
    # Check if train_acc exists (it should with the fix)
    has_acc = "train_acc" in history[0]
    
    fig2 = go.Figure()
    if has_acc:
        fig2.add_trace(go.Scatter(x=e, y=[h["train_acc"] for h in history], name="Train Acc", line=dict(dash='dash')))
    
    fig2.add_trace(go.Scatter(x=e, y=[h["val_f1_clean"] for h in history], name="Val Clean F1"))
    fig2.add_trace(go.Scatter(x=e, y=[h["val_f1_snr"] for h in history], name="Val Noisy F1"))
    fig2.add_trace(go.Scatter(x=e, y=[h["score"] for h in history], name="Score (Weighted)"))
    
    fig_save(fig2, os.path.join(out_dir, "curve_score.html"), os.path.join(out_dir, "curve_score.png"))

# =============================================================================
# Param groups
# =============================================================================
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

# =============================================================================
# Early stopping
# =============================================================================
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

# =============================================================================
# Training Logic
# =============================================================================
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
    train_noise_prob: float
) -> Dict[str, Any]:
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
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

        # --- Curriculum Noise Injection ---
        if train_noise and (random.random() < train_noise_prob):
            snr = random.uniform(train_snr_min, train_snr_max)
            x = add_awgn_torch(x, snr)
            f = f + torch.randn_like(f) * 0.2 

        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            if use_mixup:
                x_m, f_m, y_a, y_b, lam = mixup_batch(x, f, y, mixup_alpha, device)
                logits = model(x_m, f_m)
                la = criterion(logits.float(), y_a)
                lb = criterion(logits.float(), y_b)
                loss_full = lam * la + (1.0 - lam) * lb
            else:
                logits = model(x, f)
                loss_full = criterion(logits.float(), y)

        if not torch.isfinite(loss_full):
            continue

        loss = loss_full / accum_steps
        scaler.scale(loss).backward()

        # Track accuracy (approximate for mixup, but useful for sanity check)
        # Using raw logits for accuracy to see if model is learning basic patterns
        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            # If mixup, compare to y_a (dominant label usually)
            target = y_a if use_mixup else y
            correct += int((pred == target).sum().item())
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
            if scheduler is not None:
                scheduler.step()
        
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
        if not torch.isfinite(logits).all(): continue
        loss = criterion(logits.float(), y)
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
        L = tta_logits(model, x, f) if use_tta else model(x, f)
        if not torch.isfinite(L).all(): continue
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
                if torch.isfinite(L).all():
                    Lsum = L if Lsum is None else (Lsum + L)
                    valid += 1
            except: pass
        if valid == 0:
            try:
                L = models[0](x, f).float()
                L = torch.nan_to_num(L)
                Lsum = L; valid = 1
            except: continue
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
    opt.step(closure)
    return float(torch.exp(logT).item())

# =============================================================================
# Run One Seed
# =============================================================================
def run_one_seed(args, seed, data, class_names, K):
    seed_everything(seed)
    device = torch.device(args.device)
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
        w = 1.0 / counts
        sampler = WeightedRandomSampler(torch.from_numpy(w[ytr]).double(), len(ytr))
    
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=args.pin_memory)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    
    module = import_model_module(args.model_py)
    ModelCls = pick_model_class(module, args.model_class)
    feat_dim = int(data["X_train_feat"].shape[1])
    model = ModelCls(in_channels=3, num_classes=K, feat_dim=feat_dim).to(device)
    
    weight_t = None
    if args.use_class_weights:
        counts = np.bincount(ytr, minlength=K).astype(np.float64); counts[counts==0]=1.0
        w = counts.sum() / counts; w = w/w.mean()
        weight_t = torch.tensor(w, dtype=torch.float32, device=device)
    
    criterion = nn.CrossEntropyLoss(weight=weight_t, label_smoothing=args.label_smoothing)
    groups, _ = build_param_groups(model, args.lr, args.alpha_lr_scale, args.weight_decay)
    optimizer = optim.AdamW(groups)
    
    total_steps = int(len(dl_tr) * args.epochs / args.accum_steps)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[g['lr'] for g in groups], total_steps=total_steps, pct_start=args.pct_start, div_factor=args.div_factor, final_div_factor=args.final_div_factor)
    
    run_dir = ensure_dir(os.path.join(args.out_dir, f"run_{now_stamp()}_seed{seed}"))
    best_path = os.path.join(run_dir, "best_model.pth")
    save_json(vars(args)|{"seed":seed}, os.path.join(run_dir, "config.json"))
    
    best_score = -1e9
    history = []
    early = EarlyStopping(args.patience, args.min_delta)
    
    for ep in range(1, args.epochs+1):
        t0 = time.time()
        
        # 1. Warmup Logic for Noise:
        # Don't apply noise in first 5 epochs to let model learn basic patterns
        noise_active = args.train_noise and (ep > args.mixup_warmup_epochs) 
        
        # 2. Mixup Logic
        use_mix = args.mixup and (ep > args.mixup_warmup_epochs)
        
        tr = train_one_epoch(
            model, dl_tr, optimizer, scheduler, device, criterion, 
            args.amp, args.accum_steps, args.grad_clip, use_mix, args.mixup_alpha,
            noise_active, args.train_snr_min, args.train_snr_max, args.train_noise_prob
        )
        
        vL, vy = eval_split_logits(model, dl_va, device, False, None)
        f1_c = macro_f1(vy, vL.argmax(1), K) if len(vy) else 0.0
        
        f1_s = f1_c
        if args.val_snr is not None:
            vLs, vys = eval_split_logits(model, dl_va, device, False, float(args.val_snr))
            f1_s = macro_f1(vys, vLs.argmax(1), K) if len(vys) else f1_c
            
        score = args.w_clean*f1_c + args.w_snr*f1_s
        val_loss = eval_val_loss(model, dl_va, device, criterion)
        
        lrs = [g['lr'] for g in optimizer.param_groups]
        
        # FIX: Standardized keys for plotter
        row = {
            "epoch": ep, 
            "train_loss": tr['loss'], 
            "train_acc": tr['acc'],  # Added this
            "val_loss": val_loss, 
            "val_f1_clean": f1_c, 
            "val_f1_snr": f1_s, 
            "score": score, 
            "lr_min": min(lrs),
            "lr_max": max(lrs)
        }
        history.append(row)
        
        LOG.info(f"Ep {ep:02d} | L_tr={tr['loss']:.4f} Acc_tr={tr['acc']:.4f} | F1_c={f1_c:.4f} F1_s={f1_s:.4f} | Sc={score:.4f} | {time.time()-t0:.1f}s")
        
        if score > best_score:
            best_score = score
            torch.save({"model":model.state_dict(), "best_score":best_score}, best_path)
            LOG.info(f"  [BEST] {best_score:.4f}")
            
        if early.step(score):
            LOG.info("Early stopping")
            break
            
    plot_curves(history, run_dir)
    
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    # Test
    tL, ty = eval_split_logits(model, dl_te, device, (args.tta and args.tta_final_only))
    acc = float((tL.argmax(1)==ty).mean())
    f1 = macro_f1(ty, tL.argmax(1), K)
    
    cm = confusion_matrix(ty, tL.argmax(1), labels=list(range(K)))
    plot_confusion(cm, class_names, run_dir, f"test_seed{seed}")
    save_text(classification_report(ty, tL.argmax(1), labels=list(range(K)), target_names=class_names, zero_division=0), os.path.join(run_dir, "report.txt"))
    
    # SNR benches
    snr_res = {}
    for k in list(data.keys()):
        if k.startswith("test_snr_") and k.endswith("_sig"):
            snr = k.replace("test_snr_", "").replace("_sig", "")
            kf = k.replace("_sig", "_feat")
            d_snr = MemmapDataset(data[k], data[kf], np.asarray(data["y_test"]).astype(np.int64))
            l_snr = DataLoader(d_snr, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
            Ls, ys = eval_split_logits(model, l_snr, device, (args.tta and args.tta_final_only))
            snr_res[snr] = float((Ls.argmax(1)==ys).mean())
            
    if snr_res: plot_snr_curve(snr_res, run_dir, f"seed{seed}", "acc")
    final = {"test_acc":acc, "test_f1":f1, "snr":snr_res, "run_dir":run_dir}
    save_json(final, os.path.join(run_dir, "final.json"))
    return final, best_path

# =============================================================================
# CLI
# =============================================================================
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
    # REDUCED LR to stabilize noisy training
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
    # INCREASED PATIENCE
    p.add_argument("--patience", type=int, default=20) 
    p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--tta_final_only", type=int, default=1)
    
    # Ensemble
    p.add_argument("--do_ensemble", action="store_true")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--calib_use_val_snr", action="store_true")
    p.add_argument("--ece_bins", type=int, default=15)
    
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    
    # Noise args
    p.add_argument("--train_noise", action="store_true")
    p.add_argument("--train_snr_min", type=float, default=-6.0)
    p.add_argument("--train_snr_max", type=float, default=10.0)
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
    
    LOG.info(f"Classes: {K}")
    LOG.info(f"Names: {class_names}")
    
    results = []
    best_paths = []
    
    for s in args.seeds:
        res, bp = run_one_seed(args, s, data, class_names, K)
        results.append(res)
        best_paths.append(bp)
        
    # Ensemble
    if args.do_ensemble:
        LOG.info("\nEnsemble...")
        device = torch.device(args.device)
        ModelCls = pick_model_class(import_model_module(args.model_py), args.model_class)
        feat_dim = int(data["X_train_feat"].shape[1])
        
        models = []
        for bp in best_paths:
            m = ModelCls(in_channels=3, num_classes=K, feat_dim=feat_dim).to(device)
            m.load_state_dict(torch.load(bp, map_location=device)["model"])
            m.eval()
            models.append(m)
            
        yte = np.asarray(data["y_test"]).astype(np.int64)
        ds_te = MemmapDataset(data["X_test_sig"], data["X_test_feat"], yte)
        dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
        
        use_tta = (args.tta and args.tta_final_only)
        L, y = ensemble_logits(models, dl_te, device, use_tta)
        acc = float((L.argmax(1) == y).mean())
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
             L, y = ensemble_logits(models, dl_te, device, use_tta, float(snr))
             if len(L) > 0:
                 snr_curve[str(snr)] = float((L.argmax(1) == y).mean())
             
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






# python train_pu_real.py ^
#    --data_dir "D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\Paderborn\processed_pu_ultrapro_FINAL" ^
#    --out_dir  "D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\runs_pu_final_robust_v3" ^
#    --model_py "D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\Model\model.py" ^
#    --seeds 42 101 2024 ^
#    --epochs 50 ^
#    --batch_size 128 ^
#    --lr 1e-3 ^
#    --patience 15 ^
#    --val_snr -6 ^
#    --w_clean 0.5 ^
#    --w_snr 0.5 ^
#    --train_noise ^
#    --train_snr_min -4 ^
#    --train_noise_prob 0.5 ^
#    --do_ensemble ^
#    --calibrate
