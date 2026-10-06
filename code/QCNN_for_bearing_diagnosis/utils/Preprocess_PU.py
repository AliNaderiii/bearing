# # -*- coding: utf-8 -*-
# r"""
# Preprocess_PU_Final_UltraPro.py
# ================================================================================
# PU (Paderborn) Preprocessing - FINAL UltraPro (Leak-Free + Memmap + Paper Ready)
# Author: Ali Naderi - 2026 (Final)

# Why this version is "paper-ready":
# - Leak-free file-wise split (no window leakage)
# - Labeling aligned with PU datasheets:
#     * K0xx: Healthy (Normal)
#     * KAxx: Artificial Outer (often EDM/Drilling/Engraver)
#     * KIxx: Artificial/Real Inner (mixed in dataset)
#     * KBxx: Real/Compound faults (can include IR & OR simultaneously)
#   => KB should be treated as Real/Compound category, not purely Ball.
# - Supports 3 labeling modes:
#     1) compact8 (default): auto-remap to ONLY classes present in your data
#        (solves "some classes are missing" scientifically)
#     2) stage10: 10-class stage mapping:
#          0: Normal
#          1-3: Outer_Stage1..3
#          4-6: Inner_Stage1..3
#          7-9: Real_Stage1..3   (KB + some KI real can be mapped here if desired)
#     3) classic4: 4-class:
#          0 Normal, 1 Outer, 2 Inner, 3 RealCompound

# Output (memmaps, consistent with your training pipeline):
#   X_train_sig.npy, X_train_feat.npy, y_train.npy
#   X_val_sig.npy,   X_val_feat.npy,   y_val.npy
#   X_test_sig.npy,  X_test_feat.npy,  y_test.npy
#   classes.npy            (dict id->name)
#   scaler_mean.npy, scaler_scale.npy  (train-only StandardScaler)
#   feature_scaler.joblib  (optional)
#   signal_norm_stats.npz  (train-only x_min/x_max)
#   learned_filter_ch0.pt, ch1.pt, ch2.pt
#   manifest.json, meta.json

# Optional noisy test benches:
#   test_snr_-10_sig.npy, test_snr_-10_feat.npy, ...
#   (generated from normalized test signals + same feature extractor + same scaler)

# --------------------------------------------------------------------------------
# Recommended run (your current setup):
# python Preprocess_PU_Final_UltraPro.py ^
#   --root "D:\...\data\Paderborn" ^
#   --out_dir "D:\...\data\Paderborn\processed_pu_ultrapro_final" ^
#   --seed 42 ^
#   --length 2048 --stride 1024 ^
#   --downsample_q 6 --downsample_mode poly ^
#   --label_mode compact8 ^
#   --filter_epochs 2 ^
#   --make_noisy_bench --snr_list -10 -8 -6 -4 -2
# ================================================================================
# """

# from __future__ import annotations

# import os
# import re
# import sys
# import gc
# import json
# import math
# import argparse
# import logging
# import datetime
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple

# import numpy as np
# import scipy.io
# import scipy.signal

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import TensorDataset, DataLoader

# from sklearn.preprocessing import StandardScaler
# import joblib

# # Optional MAT v7.3 support
# try:
#     import h5py
#     _HAS_H5PY = True
# except Exception:
#     _HAS_H5PY = False


# # =============================================================================
# # Logger
# # =============================================================================
# logger = logging.getLogger("PU_PREPROCESS_FINAL")
# logger.setLevel(logging.INFO)
# if not logger.handlers:
#     ch = logging.StreamHandler(sys.stdout)
#     ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
#     logger.addHandler(ch)

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # =============================================================================
# # Small helpers
# # =============================================================================
# def ensure_dir(p: str) -> None:
#     os.makedirs(p, exist_ok=True)

# def save_json(obj: Any, path: str) -> None:
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(obj, f, indent=2, ensure_ascii=False)

# def open_memmap(path: str, shape: Tuple[int, ...], dtype: str = "float32") -> np.memmap:
#     return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)

# def set_seed(seed: int) -> np.random.RandomState:
#     import random
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     return np.random.RandomState(seed)

# def np_load_mmap(path: str) -> np.ndarray:
#     return np.load(path, mmap_mode="r")


# # =============================================================================
# # Config
# # =============================================================================
# @dataclass
# class CFG:
#     root: str
#     out_dir: str
#     seed: int = 42

#     # windowing
#     length: int = 2048
#     stride: int = 1024

#     # downsample
#     downsample_q: int = 6
#     downsample_mode: str = "poly"   # "poly" | "slice"

#     # split (file-wise)
#     train_ratio: float = 0.6
#     val_ratio: float = 0.2
#     test_ratio: float = 0.2

#     # labeling
#     label_mode: str = "compact8"    # "compact8" | "stage10" | "classic4"
#     # note: compact8 will auto-remap to present classes after preprocessing

#     # learned filter
#     use_filter: bool = True
#     filter_epochs: int = 2
#     filter_lr: float = 1e-3
#     filter_batch: int = 32
#     filter_reservoir_per_ch: int = 20000

#     # feature extraction
#     feat_dim: int = 21              # 7 per channel * 3
#     write_batch: int = 256          # windows per chunk

#     # noisy bench
#     make_noisy_bench: bool = False
#     snr_list: List[int] = None

#     # debug / limiting
#     max_files_per_class: int = 0    # 0 => disabled


# # =============================================================================
# # Robust MAT loader
# # =============================================================================
# def safe_np(x: Any) -> np.ndarray:
#     if isinstance(x, np.ndarray):
#         return x
#     try:
#         return np.asarray(x)
#     except Exception:
#         return np.array([], dtype=object)

# def load_mat_robust(path: str) -> Optional[dict]:
#     if not os.path.exists(path):
#         return None
#     try:
#         return scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
#     except NotImplementedError:
#         if _HAS_H5PY:
#             try:
#                 out = {}
#                 with h5py.File(path, "r") as f:
#                     def visit(name, node):
#                         if isinstance(node, h5py.Dataset) and np.issubdtype(node.dtype, np.number):
#                             val = node[()]
#                             if hasattr(val, "ndim") and val.ndim == 2:
#                                 val = val.T
#                             out[name] = val
#                     f.visititems(visit)
#                 return out
#             except Exception:
#                 return None
#     except Exception:
#         return None
#     return None

# def flatten_1d_numeric(a: Any) -> Optional[np.ndarray]:
#     arr = safe_np(a)
#     for _ in range(3):
#         if arr.dtype == object and arr.size == 1:
#             arr = safe_np(arr.flat[0])
#         else:
#             break
#     if arr.ndim == 2:
#         arr = arr.squeeze()
#     if arr.ndim != 1:
#         return None
#     if not np.issubdtype(arr.dtype, np.number):
#         return None
#     if arr.size < 4000:
#         return None
#     return arr.astype(np.float32)

# def recursive_collect_numeric_1d(obj: Any, key_prefix: str = "", max_depth: int = 10) -> List[Tuple[str, np.ndarray]]:
#     found: List[Tuple[str, np.ndarray]] = []
#     if max_depth <= 0:
#         return found

#     if isinstance(obj, dict):
#         for k, v in obj.items():
#             if isinstance(k, str) and not k.startswith("__"):
#                 found.extend(recursive_collect_numeric_1d(v, f"{key_prefix}.{k}", max_depth - 1))
#         return found

#     if hasattr(obj, "__dict__"):
#         for k, v in obj.__dict__.items():
#             if isinstance(k, str) and not k.startswith("_"):
#                 found.extend(recursive_collect_numeric_1d(v, f"{key_prefix}.{k}", max_depth - 1))
#         return found

#     arr = safe_np(obj)
#     if np.issubdtype(arr.dtype, np.number):
#         s = flatten_1d_numeric(arr)
#         if s is not None:
#             found.append((key_prefix, s))
#         return found

#     if arr.dtype == object:
#         if arr.size > 80:
#             return found
#         for i, item in enumerate(arr.ravel()):
#             found.extend(recursive_collect_numeric_1d(item, f"{key_prefix}[{i}]", max_depth - 1))
#         return found

#     return found

# def choose_3ch_signals(candidates: List[Tuple[str, np.ndarray]]) -> Optional[np.ndarray]:
#     """
#     Heuristic selection of 3 signals (3,L).
#     Prefers vibration/current signals if present.
#     """
#     if not candidates:
#         return None

#     priorities = {
#         "vib": 10, "y[6]": 10, "y(7)": 10, "mcs": 10,
#         "curr": 8, "phase": 8, "y[0]": 8, "y(1)": 8,
#         "torq": 5, "speed": 5
#     }

#     def score(key: str) -> int:
#         k = key.lower()
#         for pat, val in priorities.items():
#             if pat in k:
#                 return val
#         return 0

#     c_sorted = sorted(candidates, key=lambda x: (-score(x[0]), -x[1].size))
#     max_len = c_sorted[0][1].size
#     valid = [sig for _, sig in c_sorted if sig.size >= 0.8 * max_len]
#     if not valid:
#         valid = [c_sorted[0][1]]

#     selected = valid[:3]
#     while len(selected) < 3:
#         selected.append(selected[0])

#     min_len = min(s.size for s in selected)
#     return np.stack([s[:min_len] for s in selected[:3]], axis=0).astype(np.float32)


# # =============================================================================
# # Downsample
# # =============================================================================
# def downsample_3ch(sig3: np.ndarray, q: int, mode: str) -> np.ndarray:
#     """
#     sig3: (3,L) -> (3,L')
#     mode:
#       - slice: sig[:, ::q]
#       - poly : resample_poly(down=q) per channel (anti-alias)
#     """
#     if q <= 1:
#         return sig3.astype(np.float32)

#     if mode == "slice":
#         return sig3[:, ::q].astype(np.float32)

#     out = []
#     for c in range(3):
#         out.append(scipy.signal.resample_poly(sig3[c], up=1, down=q).astype(np.float32))
#     m = min(len(x) for x in out)
#     out = [x[:m] for x in out]
#     return np.stack(out, axis=0).astype(np.float32)


# # =============================================================================
# # Labeling (datasheet-aligned)
# # =============================================================================
# # stage10: 0 Normal, 1-3 Outer, 4-6 Inner, 7-9 Real/Compound
# CLASS_MAP_STAGE10 = {
#     0: "Normal",
#     1: "Outer_Stage1", 2: "Outer_Stage2", 3: "Outer_Stage3",
#     4: "Inner_Stage1", 5: "Inner_Stage2", 6: "Inner_Stage3",
#     7: "Real_Stage1",  8: "Real_Stage2",  9: "Real_Stage3",
# }

# CLASS_MAP_CLASSIC4 = {
#     0: "Normal",
#     1: "Outer",
#     2: "Inner",
#     3: "RealCompound",
# }

# def infer_label_from_path(path: str, label_mode: str) -> Tuple[Optional[str], int]:
#     """
#     Uses folder codes:
#       K0xx -> Normal
#       KAxx -> Outer (artificial)
#       KIxx -> Inner (mixed; still treated as Inner category)
#       KBxx -> Real/Compound category (not ball-only) per datasheets
#     For stage10:
#       stage inferred from two-digit number:
#         <=9 => stage1, <=19 => stage2, else stage3
#     """
#     folder = Path(path).parent.name.upper()
#     full = f"{folder}_{os.path.basename(path).upper()}"

#     # Healthy
#     if folder.startswith("K0"):
#         if label_mode == "classic4":
#             return "Normal", 0
#         return "Normal", 0

#     m = re.search(r"(KA|KI|KB)(\d{2})", full)
#     if not m:
#         return None, -1

#     t, n = m.group(1), int(m.group(2))
#     stage = 1 if n <= 9 else (2 if n <= 19 else 3)

#     if label_mode == "classic4":
#         if t == "KA":
#             return "Outer", 1
#         if t == "KI":
#             return "Inner", 2
#         if t == "KB":
#             return "RealCompound", 3
#         return None, -1

#     # stage10 or compact8 base-labels
#     if t == "KA":
#         cid = 1 + (stage - 1)  # 1..3
#         return f"Outer_Stage{stage}", cid
#     if t == "KI":
#         cid = 4 + (stage - 1)  # 4..6
#         return f"Inner_Stage{stage}", cid
#     if t == "KB":
#         # Real/Compound
#         cid = 7 + (stage - 1)  # 7..9
#         return f"Real_Stage{stage}", cid

#     return None, -1


# # =============================================================================
# # Learned filter (train-only)
# # =============================================================================
# class LearnedFilter(nn.Module):
#     """
#     conv + spectral gate (train-only)
#     """
#     def __init__(self, length: int):
#         super().__init__()
#         self.conv = nn.Conv1d(1, 1, 15, padding=7, bias=False)
#         self.act = nn.PReLU()
#         self.n_freq = length // 2 + 1
#         self.gate = nn.Linear(self.n_freq, self.n_freq)
#         nn.init.dirac_(self.conv.weight)
#         nn.init.constant_(self.gate.weight, 0.0)
#         nn.init.constant_(self.gate.bias, 1.0)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         xt = self.act(self.conv(x)).squeeze(1)            # (B,L)
#         X = torch.fft.rfft(xt)                            # (B,F)
#         g = torch.sigmoid(self.gate(torch.abs(X)))        # (B,F)
#         g = torch.clamp(g, 0.01, 0.99)
#         out = torch.fft.irfft(X * g, n=x.shape[-1])       # (B,L)
#         return out.unsqueeze(1)


# def train_filter_mse(model: LearnedFilter, data_1ch: np.ndarray, epochs: int, lr: float, batch: int, seed: int) -> LearnedFilter:
#     if len(data_1ch) == 0:
#         return model.eval()

#     model = model.to(DEVICE).train()
#     opt = torch.optim.Adam(model.parameters(), lr=lr)
#     g = torch.Generator().manual_seed(int(seed))

#     ds = TensorDataset(torch.from_numpy(data_1ch).float().unsqueeze(1))  # (N,1,L)
#     dl = DataLoader(ds, batch_size=batch, shuffle=True, generator=g, num_workers=0)

#     for ep in range(epochs):
#         total = 0.0
#         n = 0
#         for (xb,) in dl:
#             xb = xb.to(DEVICE)
#             out = model(xb)
#             loss = F.mse_loss(out, xb)
#             opt.zero_grad(set_to_none=True)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
#             opt.step()
#             total += float(loss.item()) * xb.size(0)
#             n += xb.size(0)
#         logger.info(f"  filter ep {ep+1}/{epochs} | loss={total/max(1,n):.6f}")
#     model.eval()
#     return model


# # =============================================================================
# # Feature extractor (7 per channel => 21)
# # =============================================================================
# class GPUFeatureExtractor(nn.Module):
#     def __init__(self, length: int, device: torch.device):
#         super().__init__()
#         self.length = length
#         self.device = device
#         self.nperseg = 256
#         self.hop = 64
#         self.register_buffer("win", torch.hann_window(self.nperseg), persistent=False)

#     @torch.no_grad()
#     def extract(self, x_tensor: torch.Tensor) -> torch.Tensor:
#         """
#         x_tensor: (B,L)
#         returns: (B,7): mu, sd, rms, peak, crest, kurt, spec_en
#         """
#         x = x_tensor.to(self.device)
#         w = self.win.to(x.device)

#         mu = torch.mean(x, dim=1, keepdim=True)
#         sd = torch.std(x, dim=1, keepdim=True) + 1e-9
#         rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-12)
#         peak = torch.max(torch.abs(x), dim=1, keepdim=True)[0]
#         crest = peak / (rms + 1e-9)
#         kurt = torch.mean((x - mu) ** 4, dim=1, keepdim=True) / (sd ** 4 + 1e-9)

#         x_pad = F.pad(x, (0, self.nperseg))
#         frames = x_pad.unfold(1, self.nperseg, self.hop) * w
#         spec = torch.fft.rfft(frames, dim=2).abs()
#         spec_en = torch.mean(torch.sum(spec ** 2, dim=2), dim=1, keepdim=True)

#         return torch.cat([mu, sd, rms, peak, crest, kurt, spec_en], dim=1)


# # =============================================================================
# # Noise bench
# # =============================================================================
# def add_wgn_np(x: np.ndarray, snr_db: float, rng: np.random.RandomState) -> np.ndarray:
#     sig_pwr = np.mean(x ** 2, axis=2, keepdims=True) + 1e-12
#     snr_lin = 10.0 ** (snr_db / 10.0)
#     noise_pwr = sig_pwr / (snr_lin + 1e-12)
#     noise = rng.randn(*x.shape).astype(np.float32) * np.sqrt(noise_pwr).astype(np.float32)
#     return (x + noise).astype(np.float32)


# # =============================================================================
# # Core pipeline
# # =============================================================================
# def scan_mat_files(root: str) -> List[str]:
#     files = []
#     for dp, _, fn in os.walk(root):
#         for f in fn:
#             if f.lower().endswith(".mat"):
#                 files.append(os.path.join(dp, f))
#     files.sort()
#     return files


# def stratified_file_split(items: List[Dict[str, Any]], seed: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, List[Dict[str, Any]]]:
#     """
#     File-wise, but stratified by label as much as possible.
#     (If a class has very few files, splits may still miss it.)
#     """
#     assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
#     rng = np.random.RandomState(seed)

#     by_label: Dict[int, List[Dict[str, Any]]] = {}
#     for it in items:
#         by_label.setdefault(it["label"], []).append(it)

#     splits = {"train": [], "val": [], "test": []}
#     for lbl, arr in by_label.items():
#         arr = list(arr)
#         rng.shuffle(arr)
#         n = len(arr)
#         n_tr = int(round(train_ratio * n))
#         n_va = int(round(val_ratio * n))
#         n_tr = max(1, min(n_tr, n - 2)) if n >= 3 else max(1, n_tr)
#         n_va = max(1, min(n_va, n - n_tr - 1)) if n - n_tr >= 2 else max(0, n_va)

#         tr = arr[:n_tr]
#         va = arr[n_tr:n_tr + n_va]
#         te = arr[n_tr + n_va:]
#         splits["train"].extend(tr)
#         splits["val"].extend(va)
#         splits["test"].extend(te)

#     # shuffle within split to remove label blocks
#     for sp in ["train","val","test"]:
#         rng.shuffle(splits[sp])
#     return splits


# def extract_windows(sig3: np.ndarray, cfg: CFG) -> np.ndarray:
#     L = sig3.shape[1]
#     if L < cfg.length:
#         return np.zeros((0, 3, cfg.length), dtype=np.float32)
#     starts = range(0, L - cfg.length + 1, cfg.stride)
#     wins = [sig3[:, s:s+cfg.length] for s in starts]
#     if not wins:
#         return np.zeros((0, 3, cfg.length), dtype=np.float32)
#     return np.stack(wins, axis=0).astype(np.float32)


# def count_windows_for_file(path: str, cfg: CFG) -> int:
#     md = load_mat_robust(path)
#     if md is None:
#         return 0
#     cands = recursive_collect_numeric_1d(md)
#     sig = choose_3ch_signals(cands)
#     if sig is None:
#         return 0
#     sig_ds = downsample_3ch(sig, cfg.downsample_q, cfg.downsample_mode)
#     L = sig_ds.shape[1]
#     if L < cfg.length:
#         return 0
#     return (L - cfg.length) // cfg.stride + 1


# def remap_compact_labels(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, class_map: Dict[int, str]):
#     present = sorted(set(np.unique(y_train).tolist()) | set(np.unique(y_val).tolist()) | set(np.unique(y_test).tolist()))
#     old2new = {old: i for i, old in enumerate(present)}
#     new2name = {old2new[old]: class_map[old] for old in present}
#     ytr2 = np.vectorize(old2new.get)(y_train).astype(np.int64)
#     yva2 = np.vectorize(old2new.get)(y_val).astype(np.int64)
#     yte2 = np.vectorize(old2new.get)(y_test).astype(np.int64)
#     return ytr2, yva2, yte2, old2new, new2name


# def main(cfg: CFG) -> None:
#     ensure_dir(cfg.out_dir)
#     rng = set_seed(cfg.seed)

#     logger.info("======================================================================")
#     logger.info("PU FINAL PREPROCESS START")
#     logger.info(f"Device: {DEVICE}")
#     logger.info(f"ROOT:   {cfg.root}")
#     logger.info(f"OUT:    {cfg.out_dir}")
#     logger.info(f"WIN:    length={cfg.length}, stride={cfg.stride}")
#     logger.info(f"DS:     q={cfg.downsample_q}, mode={cfg.downsample_mode}")
#     logger.info(f"SPLIT:  train/val/test={cfg.train_ratio}/{cfg.val_ratio}/{cfg.test_ratio} (file-wise stratified)")
#     logger.info(f"LABEL:  mode={cfg.label_mode}")
#     logger.info(f"FILTER: use={cfg.use_filter} epochs={cfg.filter_epochs}")
#     logger.info("======================================================================")

#     # 1) Scan files
#     all_files = scan_mat_files(cfg.root)
#     if not all_files:
#         raise FileNotFoundError("No .mat files found.")
#     logger.info(f"Files scanned: {len(all_files)}")

#     # 2) Label files
#     items = []
#     skipped = 0
#     for p in all_files:
#         lbl_str, lbl = infer_label_from_path(p, cfg.label_mode if cfg.label_mode != "compact8" else "stage10")
#         # compact8 uses stage10 base labels then remaps later
#         if lbl < 0:
#             skipped += 1
#             continue
#         items.append({"path": p, "label": int(lbl), "label_str": lbl_str})

#     logger.info(f"Labeled: {len(items)} | skipped(no label): {skipped}")
#     if len(items) == 0:
#         raise RuntimeError("No labeled files found.")

#     # Optional: cap files per class for debugging
#     if cfg.max_files_per_class and cfg.max_files_per_class > 0:
#         byc: Dict[int, List[Dict[str, Any]]] = {}
#         for it in items:
#             byc.setdefault(it["label"], []).append(it)
#         items2 = []
#         for c in sorted(byc):
#             items2.extend(byc[c][:cfg.max_files_per_class])
#         items = items2
#         logger.warning(f"Applied max_files_per_class={cfg.max_files_per_class}. Remaining={len(items)}")

#     # 3) File-wise stratified split
#     splits = stratified_file_split(items, cfg.seed, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio)
#     logger.info(f"Split file counts: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

#     # 4) Count windows per split (exact method)
#     logger.info("Counting windows (exact downsample)...")
#     counts = {"train": 0, "val": 0, "test": 0}
#     kept = {"train": [], "val": [], "test": []}

#     for sp in ["train","val","test"]:
#         for it in splits[sp]:
#             nwin = count_windows_for_file(it["path"], cfg)
#             if nwin > 0:
#                 it2 = dict(it)
#                 it2["n_win"] = int(nwin)
#                 kept[sp].append(it2)
#                 counts[sp] += int(nwin)

#     logger.info(f"Window counts: {counts}")
#     if counts["train"] == 0:
#         raise RuntimeError("Zero training windows. Check downsample_q/length/stride.")

#     # 5) Allocate memmaps
#     mms: Dict[str, np.memmap] = {}
#     for sp in ["train","val","test"]:
#         N = counts[sp]
#         if N <= 0:
#             continue
#         mms[f"{sp}_sig"] = open_memmap(os.path.join(cfg.out_dir, f"X_{sp}_sig.npy"), (N, 3, cfg.length), dtype="float32")
#         mms[f"{sp}_feat"] = open_memmap(os.path.join(cfg.out_dir, f"X_{sp}_feat.npy"), (N, cfg.feat_dim), dtype="float32")
#         mms[f"{sp}_y"] = open_memmap(os.path.join(cfg.out_dir, f"y_{sp}.npy"), (N,), dtype="int64")

#     # class map baseline
#     class_map_base = CLASS_MAP_STAGE10 if cfg.label_mode in ("stage10","compact8") else CLASS_MAP_CLASSIC4

#     # write initial classes.npy (may be overwritten if compact8 remap)
#     np.save(os.path.join(cfg.out_dir, "classes.npy"), np.array([class_map_base], dtype=object), allow_pickle=True)

#     # 6) Train learned filters (train-only)
#     filters = [LearnedFilter(cfg.length).to(DEVICE) for _ in range(3)]
#     if cfg.use_filter:
#         logger.info("Building reservoir for filter training (train-only)...")
#         res_ch = [[], [], []]
#         seen = 0
#         for it in kept["train"]:
#             md = load_mat_robust(it["path"])
#             if md is None:
#                 continue
#             sig = choose_3ch_signals(recursive_collect_numeric_1d(md))
#             if sig is None:
#                 continue
#             sig_ds = downsample_3ch(sig, cfg.downsample_q, cfg.downsample_mode)
#             wins = extract_windows(sig_ds, cfg)
#             if len(wins) == 0:
#                 continue
#             take = min(len(wins), 50)
#             idx = rng.choice(len(wins), take, replace=False) if len(wins) > take else np.arange(len(wins))
#             ws = wins[idx]
#             for c in range(3):
#                 res_ch[c].append(ws[:, c, :])
#             seen += take
#             if seen >= cfg.filter_reservoir_per_ch:
#                 break

#         for c in range(3):
#             data_1ch = np.concatenate(res_ch[c], axis=0).astype(np.float32) if len(res_ch[c]) else np.zeros((0, cfg.length), dtype=np.float32)
#             if len(data_1ch) > cfg.filter_reservoir_per_ch:
#                 idx = rng.choice(len(data_1ch), cfg.filter_reservoir_per_ch, replace=False)
#                 data_1ch = data_1ch[idx]
#             logger.info(f"Training filter ch{c} on N={len(data_1ch)} windows...")
#             filters[c] = train_filter_mse(filters[c], data_1ch, cfg.filter_epochs, cfg.filter_lr, cfg.filter_batch, cfg.seed + 17*c)

#         for c in range(3):
#             torch.save(filters[c].state_dict(), os.path.join(cfg.out_dir, f"learned_filter_ch{c}.pt"))
#         if DEVICE.type == "cuda":
#             torch.cuda.empty_cache()
#         gc.collect()
#     else:
#         logger.info("Skipping learned filter training (use_filter=False).")

#     # 7) Main processing: filter -> features -> write memmaps
#     extractor = GPUFeatureExtractor(cfg.length, DEVICE).eval()

#     x_min = np.full((1,3,1), np.inf, dtype=np.float32)
#     x_max = np.full((1,3,1), -np.inf, dtype=np.float32)

#     feat_scaler = StandardScaler()
#     scaler_fitted = False

#     cursor = {"train": 0, "val": 0, "test": 0}

#     logger.info("Writing memmaps (filtered signals + raw features)...")
#     for sp in ["train","val","test"]:
#         N_total = counts[sp]
#         if N_total <= 0:
#             continue
#         logger.info(f"Processing split={sp} | files={len(kept[sp])} | windows={N_total}")
#         for it in kept[sp]:
#             md = load_mat_robust(it["path"])
#             if md is None:
#                 continue
#             sig = choose_3ch_signals(recursive_collect_numeric_1d(md))
#             if sig is None:
#                 continue
#             sig_ds = downsample_3ch(sig, cfg.downsample_q, cfg.downsample_mode)
#             wins = extract_windows(sig_ds, cfg)   # (Nw,3,L)
#             if len(wins) == 0:
#                 continue

#             n_w = wins.shape[0]
#             start = 0
#             while start < n_w:
#                 end = min(n_w, start + cfg.write_batch)
#                 batch = wins[start:end]  # (B,3,L)
#                 B = batch.shape[0]

#                 bt = torch.from_numpy(batch).float().to(DEVICE)

#                 with torch.no_grad():
#                     if cfg.use_filter:
#                         out = torch.empty_like(bt)
#                         for c in range(3):
#                             xf = filters[c](bt[:, c:c+1, :])  # (B,1,L)
#                             out[:, c:c+1, :] = xf
#                         bt_f = out
#                     else:
#                         bt_f = bt

#                     feats = []
#                     for c in range(3):
#                         feats.append(extractor.extract(bt_f[:, c, :]))
#                     feats_t = torch.cat(feats, dim=1)  # (B,21)

#                 sig_np = bt_f.detach().cpu().numpy().astype(np.float32)
#                 feat_np = feats_t.detach().cpu().numpy().astype(np.float32)

#                 # update train-only stats
#                 if sp == "train":
#                     bmin = sig_np.min(axis=(0,2), keepdims=True)
#                     bmax = sig_np.max(axis=(0,2), keepdims=True)
#                     x_min = np.minimum(x_min, bmin)
#                     x_max = np.maximum(x_max, bmax)

#                     if feat_np.shape[0] > 0:
#                         feat_scaler.partial_fit(feat_np)
#                         scaler_fitted = True

#                 cur = cursor[sp]
#                 cur_end = cur + B
#                 if cur_end > counts[sp]:
#                     B = counts[sp] - cur
#                     if B <= 0:
#                         break
#                     sig_np = sig_np[:B]
#                     feat_np = feat_np[:B]
#                     cur_end = cur + B

#                 mms[f"{sp}_sig"][cur:cur_end] = sig_np
#                 mms[f"{sp}_feat"][cur:cur_end] = feat_np
#                 mms[f"{sp}_y"][cur:cur_end] = int(it["label"])

#                 cursor[sp] = cur_end
#                 start = end

#             if cursor[sp] >= counts[sp]:
#                 break

#     if not scaler_fitted:
#         raise RuntimeError("Feature scaler could not be fitted (no train features).")

#     scaler_mean = feat_scaler.mean_.astype(np.float32)
#     scaler_scale = feat_scaler.scale_.astype(np.float32)

#     np.save(os.path.join(cfg.out_dir, "scaler_mean.npy"), scaler_mean)
#     np.save(os.path.join(cfg.out_dir, "scaler_scale.npy"), scaler_scale)
#     joblib.dump(feat_scaler, os.path.join(cfg.out_dir, "feature_scaler.joblib"))

#     np.savez(os.path.join(cfg.out_dir, "signal_norm_stats.npz"), x_min=x_min, x_max=x_max)
#     logger.info("Saved train-only scaler_mean/scaler_scale and signal_norm_stats.")

#     # 8) Normalize in-place
#     denom = (x_max - x_min).copy()
#     denom[denom == 0] = 1e-6

#     logger.info("In-place signal normalization to [-1,1] (train-only min/max) ...")
#     def norm_sig(mm: np.memmap, N: int):
#         bs = 4096
#         for i in range(0, N, bs):
#             chunk = mm[i:i+bs]
#             chunk = (2.0 * (chunk - x_min) / denom - 1.0).astype(np.float32)
#             mm[i:i+bs] = chunk

#     for sp in ["train","val","test"]:
#         if counts[sp] > 0:
#             norm_sig(mms[f"{sp}_sig"], counts[sp])

#     logger.info("In-place feature standardization (train-only scaler) ...")
#     def norm_feat(mm: np.memmap, N: int):
#         bs = 8192
#         m = scaler_mean.reshape(1, -1)
#         s = scaler_scale.reshape(1, -1)
#         s = np.where(s == 0, 1.0, s)
#         for i in range(0, N, bs):
#             chunk = mm[i:i+bs].astype(np.float32)
#             chunk = (chunk - m) / s
#             mm[i:i+bs] = chunk.astype(np.float32)

#     for sp in ["train","val","test"]:
#         if counts[sp] > 0:
#             norm_feat(mms[f"{sp}_feat"], counts[sp])

#     # 9) Optional noisy test benches (RAM-safe chunking)
#     if cfg.make_noisy_bench and cfg.snr_list:
#         logger.info(f"Generating noisy benchmarks for test SNR list: {cfg.snr_list}")
#         N = counts["test"]
#         if N > 0:
#             for snr in cfg.snr_list:
#                 out_sig = open_memmap(os.path.join(cfg.out_dir, f"test_snr_{snr}_sig.npy"), (N, 3, cfg.length), dtype="float32")
#                 out_feat = open_memmap(os.path.join(cfg.out_dir, f"test_snr_{snr}_feat.npy"), (N, cfg.feat_dim), dtype="float32")

#                 bs = 2048
#                 m = scaler_mean.reshape(1, -1)
#                 s = np.where(scaler_scale.reshape(1, -1) == 0, 1.0, scaler_scale.reshape(1, -1))

#                 for i in range(0, N, bs):
#                     chunk = mms["test_sig"][i:i+bs].astype(np.float32)  # already normalized
#                     noisy = add_wgn_np(chunk, float(snr), rng)          # keep in normalized domain

#                     bt = torch.from_numpy(noisy).float().to(DEVICE)
#                     with torch.no_grad():
#                         feats = []
#                         for c in range(3):
#                             feats.append(extractor.extract(bt[:, c, :]))
#                         feats_t = torch.cat(feats, dim=1)
#                     feat_np = feats_t.cpu().numpy().astype(np.float32)
#                     feat_np = (feat_np - m) / s

#                     out_sig[i:i+noisy.shape[0]] = noisy
#                     out_feat[i:i+feat_np.shape[0]] = feat_np

#                 logger.info(f"Saved benchmark SNR={snr} memmaps.")

#     # 10) Optional compact remap (recommended if some classes missing)
#     old2new = None
#     new_class_map = None

#     if cfg.label_mode == "compact8":
#         # load y from memmaps (small)
#         y_train = np.array(mms["train_y"], dtype=np.int64)
#         y_val   = np.array(mms["val_y"], dtype=np.int64) if counts["val"] > 0 else np.zeros((0,), dtype=np.int64)
#         y_test  = np.array(mms["test_y"], dtype=np.int64) if counts["test"] > 0 else np.zeros((0,), dtype=np.int64)

#         ytr2, yva2, yte2, old2new, new_class_map = remap_compact_labels(y_train, y_val, y_test, CLASS_MAP_STAGE10)

#         # overwrite y memmaps
#         mms["train_y"][:] = ytr2
#         if counts["val"] > 0:  mms["val_y"][:] = yva2
#         if counts["test"] > 0: mms["test_y"][:] = yte2

#         # overwrite classes.npy
#         np.save(os.path.join(cfg.out_dir, "classes.npy"), np.array([new_class_map], dtype=object), allow_pickle=True)

#         logger.info(f"[COMPACT REMAP] present old classes: {sorted(old2new.keys())}")
#         logger.info(f"[COMPACT REMAP] new classes: {new_class_map}")

#     # 11) Save meta/manifest
#     # class distributions
#     def bincount_safe(y: np.ndarray, K: int) -> List[int]:
#         if len(y) == 0:
#             return [0]*K
#         return np.bincount(y.astype(np.int64), minlength=K).tolist()

#     # determine final num_classes
#     if cfg.label_mode == "compact8" and new_class_map is not None:
#         num_classes_final = len(new_class_map)
#         class_map_final = new_class_map
#     elif cfg.label_mode == "classic4":
#         num_classes_final = 4
#         class_map_final = CLASS_MAP_CLASSIC4
#     else:
#         num_classes_final = 10
#         class_map_final = CLASS_MAP_STAGE10

#     # y arrays (small)
#     ytr = np.array(mms["train_y"], dtype=np.int64)
#     yva = np.array(mms["val_y"], dtype=np.int64) if counts["val"] > 0 else np.zeros((0,), dtype=np.int64)
#     yte = np.array(mms["test_y"], dtype=np.int64) if counts["test"] > 0 else np.zeros((0,), dtype=np.int64)

#     meta = {
#         "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
#         "device": str(DEVICE),
#         "cfg": cfg.__dict__,
#         "counts_windows": counts,
#         "cursor_final": cursor,
#         "label_mode": cfg.label_mode,
#         "num_classes": int(num_classes_final),
#         "class_map": {str(k): v for k, v in class_map_final.items()},
#         "class_counts": {
#             "train": bincount_safe(ytr, num_classes_final),
#             "val": bincount_safe(yva, num_classes_final),
#             "test": bincount_safe(yte, num_classes_final),
#         },
#         "old2new_class_map": {str(k): int(v) for k, v in old2new.items()} if old2new else None,
#     }
#     save_json(meta, os.path.join(cfg.out_dir, "meta.json"))

#     manifest = {
#         "created_at": meta["created_at"],
#         "label_mode": cfg.label_mode,
#         "files": {
#             sp: [{"path": it["path"], "label": it["label"], "label_str": it["label_str"], "n_win": it["n_win"]}
#                  for it in kept[sp]]
#             for sp in ["train","val","test"]
#         }
#     }
#     save_json(manifest, os.path.join(cfg.out_dir, "manifest.json"))

#     logger.info(f"Final cursors: {cursor} vs counts={counts}")
#     for sp in ["train","val","test"]:
#         if cursor[sp] != counts[sp]:
#             logger.warning(f"[WARN] cursor mismatch on {sp}: cursor={cursor[sp]} count={counts[sp]} (some files skipped)")
#     logger.info("DONE.")


# # =============================================================================
# # CLI
# # =============================================================================
# def parse_args() -> CFG:
#     p = argparse.ArgumentParser()
#     p.add_argument("--root", type=str, required=True)
#     p.add_argument("--out_dir", type=str, required=True)
#     p.add_argument("--seed", type=int, default=42)

#     p.add_argument("--length", type=int, default=2048)
#     p.add_argument("--stride", type=int, default=1024)

#     p.add_argument("--downsample_q", type=int, default=6)
#     p.add_argument("--downsample_mode", type=str, default="poly", choices=["poly","slice"])

#     p.add_argument("--train_ratio", type=float, default=0.6)
#     p.add_argument("--val_ratio", type=float, default=0.2)
#     p.add_argument("--test_ratio", type=float, default=0.2)

#     p.add_argument("--label_mode", type=str, default="compact8", choices=["compact8","stage10","classic4"])

#     p.add_argument("--no_filter", action="store_true")
#     p.add_argument("--filter_epochs", type=int, default=2)
#     p.add_argument("--filter_lr", type=float, default=1e-3)
#     p.add_argument("--filter_batch", type=int, default=32)
#     p.add_argument("--filter_reservoir_per_ch", type=int, default=20000)

#     p.add_argument("--write_batch", type=int, default=256)

#     p.add_argument("--make_noisy_bench", action="store_true")
#     p.add_argument("--snr_list", nargs="+", type=int, default=[-10, -8, -6, -4, -2])

#     p.add_argument("--max_files_per_class", type=int, default=0)

#     a = p.parse_args()

#     cfg = CFG(
#         root=a.root,
#         out_dir=a.out_dir,
#         seed=a.seed,
#         length=a.length,
#         stride=a.stride,
#         downsample_q=a.downsample_q,
#         downsample_mode=a.downsample_mode,
#         train_ratio=a.train_ratio,
#         val_ratio=a.val_ratio,
#         test_ratio=a.test_ratio,
#         label_mode=a.label_mode,
#         use_filter=(not a.no_filter),
#         filter_epochs=a.filter_epochs,
#         filter_lr=a.filter_lr,
#         filter_batch=a.filter_batch,
#         filter_reservoir_per_ch=a.filter_reservoir_per_ch,
#         write_batch=a.write_batch,
#         make_noisy_bench=a.make_noisy_bench,
#         snr_list=a.snr_list,
#         max_files_per_class=a.max_files_per_class,
#     )
#     return cfg


# if __name__ == "__main__":
#     logger.info(f"Computation Device: {DEVICE}")
#     cfg = parse_args()
#     main(cfg)





























# -*- coding: utf-8 -*-
r"""
Preprocess_PU_Final_UltraPro_v2.py
=================================
PU (Paderborn) Preprocessing — FINAL UltraPro (Paper-ready + Memmap + Leak-free)

Author: Ali Naderi (2026) — Revised (AI-assisted)
Purpose:
  - File-wise stratified split (NO window leakage)
  - Train-only learned filters (per-channel)
  - Train-only feature scaler (StandardScaler)
  - Signal normalization using train min/max (in-place memmap)
  - Memory-efficient memmap writing (RAM-friendly)
  - Optional noisy test benches (SNRs: -10, -8, -6, -4, -2 default)
  - Output files & meta compatible with train pipelines:
      X_train_sig.npy, X_train_feat.npy, y_train.npy
      X_val_sig.npy,   X_val_feat.npy,   y_val.npy
      X_test_sig.npy,  X_test_feat.npy,  y_test.npy
      classes.npy, meta.json, manifest.json
      scaler_mean.npy, scaler_scale.npy, feature_scaler.joblib
      learned_filter_ch0.pt / ch1 / ch2  (state_dict)
Notes:
  - Designed to work with limited RAM (memmap + chunking)
  - All train-derived statistics computed using TRAIN split only
  - Strictly avoids any leakage between train/val/test
  - Robust MAT loader with h5py fallback
"""

from __future__ import annotations

import os
import sys
import re
import json
import math
import argparse
import logging
import datetime
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.signal
import scipy.io

# torch used for learned filters + GPU accel (optional)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
import joblib

# Optional MAT v7.3 support
try:
    import h5py
    _HAS_H5PY = True
except Exception:
    _HAS_H5PY = False

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("PU_PREPROCESS_FINAL")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)

# Device (used for filter training / feature extraction speedups)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Configuration dataclass
# =========================
@dataclass
class CFG:
    root: str
    out_dir: str
    seed: int = 42

    # windowing
    length: int = 2048
    stride: int = 1024

    # downsample
    downsample_q: int = 6
    downsample_mode: str = "poly"  # "poly" (anti-alias) | "slice" (fast)

    # split (file-wise)
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2

    # labeling
    label_mode: str = "compact8"    # "compact8" | "stage10" | "classic4"

    # learned filter (train-only)
    use_filter: bool = True
    filter_epochs: int = 3
    filter_lr: float = 1e-3
    filter_batch: int = 64
    filter_reservoir_per_ch: int = 20000
    filter_taps: int = 31

    # feature extraction
    feat_dim: int = 21              # default 7 per channel => 21
    write_batch: int = 256          # windows per chunk write
    mmap_write_batch: int = 4096    # chunk size for memmap in-place normalization

    # noisy bench
    make_noisy_bench: bool = False
    snr_list: List[int] = None

    # debug / limiting
    max_files_per_class: int = 0    # 0 => disabled

# =========================
# CLASS MAPS
# =========================
CLASS_MAP_STAGE10 = {
    0: "Normal",
    1: "Outer_Stage1", 2: "Outer_Stage2", 3: "Outer_Stage3",
    4: "Inner_Stage1", 5: "Inner_Stage2", 6: "Inner_Stage3",
    7: "Real_Stage1",  8: "Real_Stage2",  9: "Real_Stage3",
}

CLASS_MAP_CLASSIC4 = {
    0: "Normal",
    1: "Outer",
    2: "Inner",
    3: "RealCompound",
}

# =========================
# File / MAT utilities
# =========================
def safe_np(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.asarray(x)
    except Exception:
        return np.array([], dtype=object)

def load_mat_robust(path: str) -> Optional[dict]:
    """
    Robust loader: try scipy.io.loadmat, fallback to h5py for v7.3 mats.
    Returns dict-like mapping names->arrays for numeric datasets.
    """
    if not os.path.exists(path):
        return None
    try:
        return scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
    except NotImplementedError:
        # likely v7.3 HDF5 MAT file
        if _HAS_H5PY:
            out = {}
            try:
                with h5py.File(path, "r") as f:
                    def visit(name, node):
                        if isinstance(node, h5py.Dataset):
                            try:
                                val = node[()]
                                out[name] = val
                            except Exception:
                                pass
                    f.visititems(visit)
                return out
            except Exception:
                return None
    except Exception:
        return None
    return None

def flatten_1d_numeric(a: Any) -> Optional[np.ndarray]:
    arr = safe_np(a)
    # unwrap object arrays nested
    for _ in range(4):
        if arr.dtype == object and arr.size == 1:
            arr = safe_np(arr.flat[0])
        else:
            break
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.squeeze()
    if arr.ndim != 1:
        return None
    if not np.issubdtype(arr.dtype, np.number):
        return None
    if arr.size < 4000:
        # too short to be useful for long-window extraction
        return None
    return arr.astype(np.float32)

def recursive_collect_numeric_1d(obj: Any, key_prefix: str = "", max_depth: int = 8) -> List[Tuple[str, np.ndarray]]:
    found: List[Tuple[str, np.ndarray]] = []
    if max_depth <= 0:
        return found

    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and not k.startswith("__"):
                found.extend(recursive_collect_numeric_1d(v, f"{key_prefix}.{k}", max_depth - 1))
        return found

    if hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            if isinstance(k, str) and not k.startswith("_"):
                found.extend(recursive_collect_numeric_1d(v, f"{key_prefix}.{k}", max_depth - 1))
        return found

    arr = safe_np(obj)
    if np.issubdtype(arr.dtype, np.number):
        s = flatten_1d_numeric(arr)
        if s is not None:
            found.append((key_prefix, s))
        return found

    if arr.dtype == object:
        if arr.size > 120:
            return found
        for i, item in enumerate(arr.ravel()):
            found.extend(recursive_collect_numeric_1d(item, f"{key_prefix}[{i}]", max_depth - 1))
        return found

    return found

def choose_3ch_signals(candidates: List[Tuple[str, np.ndarray]]) -> Optional[np.ndarray]:
    """
    Heuristic to select 3 channels from candidate 1D numeric arrays.
    Returns (3, L) float32 or None.
    """
    if not candidates:
        return None

    # scoring by keyword & length
    priorities = {"vib": 10, "vibration": 10, "acc": 10, "mcs": 10, "curr": 7, "phase": 6, "torq": 3}
    scored = []
    for k, arr in candidates:
        key = (k or "").lower()
        score = 0
        for pat, v in priorities.items():
            if pat in key:
                score += v
        scored.append((score, arr.size, k, arr))

    scored.sort(key=lambda x: (-x[0], -x[1]))
    arrays = [t[3] for t in scored]

    # pick top 3 by length (pad/trim later)
    selected = arrays[:3] if len(arrays) >= 3 else arrays + [arrays[0]] * (3 - len(arrays))
    min_len = min(s.size for s in selected)
    stacked = np.stack([s[:min_len] for s in selected[:3]], axis=0).astype(np.float32)
    return stacked

# =========================
# Downsample utilities
# =========================
def downsample_3ch(sig3: np.ndarray, q: int, mode: str) -> np.ndarray:
    """
    sig3: (3, L) -> (3, L')
    mode: 'slice' (fast) or 'poly' (resample_poly anti-alias)
    """
    if q <= 1:
        return sig3.astype(np.float32)
    if mode == "slice":
        return sig3[:, ::q].astype(np.float32)
    out = []
    for c in range(3):
        y = scipy.signal.resample_poly(sig3[c], up=1, down=q).astype(np.float32)
        out.append(y)
    m = min(len(x) for x in out)
    out = [x[:m] for x in out]
    return np.stack(out, axis=0).astype(np.float32)

# =========================
# Window extraction
# =========================
def extract_windows(sig3: np.ndarray, length: int, stride: int) -> np.ndarray:
    L = sig3.shape[1]
    if L < length:
        return np.zeros((0, 3, length), dtype=np.float32)
    starts = range(0, L - length + 1, stride)
    wins = [sig3[:, s:s+length] for s in starts]
    if not wins:
        return np.zeros((0, 3, length), dtype=np.float32)
    return np.stack(wins, axis=0).astype(np.float32)

# =========================
# Learned filter model
# =========================
class LearnedFilter(nn.Module):
    """
    Simple learnable per-channel 1D conv + spectral gating.
    Trained with MSE to reconstruct clean windows (denoising-like auto-target).
    Trained ONLY on TRAIN windows (no leakage).
    """
    def __init__(self, length: int, taps: int = 31):
        super().__init__()
        self.taps = taps
        # small conv initialized with dirac-ish center
        self.conv = nn.Conv1d(1, 1, kernel_size=taps, padding=taps//2, bias=False)
        nn.init.zeros_(self.conv.weight)
        center = taps // 2
        with torch.no_grad():
            self.conv.weight[0,0,center] = 1.0
        self.act = nn.PReLU()
        # spectral gate
        self.nfreq = length // 2 + 1
        self.gate = nn.Sequential(
            nn.Linear(self.nfreq, self.nfreq),
            nn.Sigmoid()
        )
        # small init: bias towards pass-through
        for m in self.gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0.0)
                nn.init.constant_(m.bias, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,L)
        xt = self.act(self.conv(x)).squeeze(1)             # (B,L)
        X = torch.fft.rfft(xt, dim=-1)                     # (B, F)
        mag = torch.abs(X)
        g = self.gate(mag)                                 # (B, F)
        out = torch.fft.irfft(X * g, n=x.shape[-1], dim=-1)  # (B, L)
        return out.unsqueeze(1)

def train_filter_mse(model: LearnedFilter, data_1ch: np.ndarray, epochs: int, lr: float, batch: int, seed: int) -> LearnedFilter:
    """
    Train filter to reconstruct windows (self-supervised MSE).
    data_1ch: (N, L) np.float32
    """
    if data_1ch.shape[0] == 0:
        model.eval()
        return model

    device = DEVICE
    model = model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    g = torch.Generator().manual_seed(int(seed))

    ds = TensorDataset(torch.from_numpy(data_1ch).float().unsqueeze(1))  # (N,1,L)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, generator=g, num_workers=0)

    for ep in range(epochs):
        tot = 0.0
        n = 0
        for (xb,) in dl:
            xb = xb.to(device)
            out = model(xb)
            loss = F.mse_loss(out, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            bs = xb.size(0)
            tot += float(loss.item()) * bs
            n += bs
        logger.info(f"  filter ep {ep+1}/{epochs} | mse={tot/max(1,n):.8f}")
    model.eval()
    return model

# =========================
# Feature extractor (GPU friendly)
# =========================
class GPUFeatureExtractor(nn.Module):
    """
    Extract per-window engineered features.
    Returns 7 features per channel (mu, sd, rms, peak, crest, kurtosis, log_spec_energy)
    """
    def __init__(self, length: int, device: torch.device):
        super().__init__()
        self.length = length
        self.device = device
        self.nperseg = 256
        self.hop = 64
        self.register_buffer("win", torch.hann_window(self.nperseg), persistent=False)

    @torch.no_grad()
    def extract(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        x_tensor: (B, L) float32 on correct device
        returns: (B, 7) tensor
        """
        x = x_tensor.to(self.device)
        mu = torch.mean(x, dim=1, keepdim=True)
        sd = torch.std(x, dim=1, keepdim=True) + 1e-9
        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-12)
        peak = torch.max(torch.abs(x), dim=1, keepdim=True)[0]
        crest = peak / (rms + 1e-9)
        kurt = torch.mean((x - mu) ** 4, dim=1, keepdim=True) / (sd ** 4 + 1e-9)

        # spectal energy (log1p of mean sum of magnitude^2)
        x_pad = F.pad(x, (0, self.nperseg))
        frames = x_pad.unfold(1, self.nperseg, self.hop) * self.win.to(x.device)
        spec = torch.fft.rfft(frames, dim=2).abs()
        spec_en = torch.mean(torch.sum(spec ** 2, dim=2), dim=1, keepdim=True)
        spec_en = torch.log1p(spec_en)

        return torch.cat([mu, sd, rms, peak, crest, kurt, spec_en], dim=1)

# =========================
# Noise utils
# =========================
def add_wgn_np(x: np.ndarray, snr_db: float, rng: np.random.RandomState) -> np.ndarray:
    """
    x: (N, C, L) in normalized domain ([-1,1] after train min/max normalization)
    Adds WGN per-sample so resulting SNR (signal power / noise power) ~ given dB
    """
    sig_pwr = np.mean(x ** 2, axis=2, keepdims=True) + 1e-12
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_pwr = sig_pwr / (snr_lin + 1e-12)
    noise = rng.randn(*x.shape).astype(np.float32) * np.sqrt(noise_pwr.astype(np.float32))
    return (x + noise).astype(np.float32)

# =========================
# Utility: memmap open safe
# =========================
def open_memmap(path: str, shape: Tuple[int, ...], dtype: str = "float32") -> np.memmap:
    dirp = os.path.dirname(path)
    os.makedirs(dirp or ".", exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)

# =========================
# LABEL infer
# =========================
def infer_label_from_path(path: str, label_mode: str) -> Tuple[Optional[str], int]:
    """
    Infer label (string, id) from file path using folder/file name heuristics.
    Uses K0xx, KAxx, KIxx, KBxx patterns.
    For compact8 uses stage10 base mapping then remap.
    """
    folder = Path(path).parent.name.upper()
    base = Path(path).name.upper()
    full = f"{folder}_{base}"

    # Healthy K0
    if folder.startswith("K0"):
        return "Normal", 0

    m = re.search(r"(KA|KI|KB)(\d{2})", full)
    if not m:
        return None, -1

    t, n = m.group(1), int(m.group(2))
    stage = 1 if n <= 9 else (2 if n <= 19 else 3)

    if label_mode == "classic4":
        if t == "KA":
            return "Outer", 1
        if t == "KI":
            return "Inner", 2
        if t == "KB":
            return "RealCompound", 3
        return None, -1

    if t == "KA":
        cid = 1 + (stage - 1)       # 1..3
        return f"Outer_Stage{stage}", cid
    if t == "KI":
        cid = 4 + (stage - 1)       # 4..6
        return f"Inner_Stage{stage}", cid
    if t == "KB":
        cid = 7 + (stage - 1)       # 7..9
        return f"Real_Stage{stage}", cid

    return None, -1

# =========================
# Compact remap helper
# =========================
def remap_compact_labels(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, class_map: Dict[int, str]):
    present = sorted(set(np.unique(y_train).tolist()) | set(np.unique(y_val).tolist()) | set(np.unique(y_test).tolist()))
    old2new = {old: i for i, old in enumerate(present)}
    new2name = {old2new[old]: class_map[old] for old in present}
    ytr2 = np.vectorize(old2new.get)(y_train).astype(np.int64)
    yva2 = np.vectorize(old2new.get)(y_val).astype(np.int64)
    yte2 = np.vectorize(old2new.get)(y_test).astype(np.int64)
    return ytr2, yva2, yte2, old2new, new2name

# =========================
# Core preprocessing main
# =========================
def main(cfg: CFG) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    rng = np.random.RandomState(cfg.seed)

    logger.info("======================================================================")
    logger.info("PU FINAL PREPROCESS (UltraPro) START")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"ROOT:   {cfg.root}")
    logger.info(f"OUT:    {cfg.out_dir}")
    logger.info(f"WIN:    length={cfg.length}, stride={cfg.stride}")
    logger.info(f"DS:     q={cfg.downsample_q}, mode={cfg.downsample_mode}")
    logger.info(f"SPLIT:  train/val/test={cfg.train_ratio}/{cfg.val_ratio}/{cfg.test_ratio} (file-wise stratified)")
    logger.info(f"LABEL:  mode={cfg.label_mode}")
    logger.info(f"FILTER: use={cfg.use_filter} epochs={cfg.filter_epochs} taps={cfg.filter_taps}")
    logger.info("======================================================================")

    # 1) scan .mat files
    all_files = []
    for dp, _, fn in os.walk(cfg.root):
        for f in fn:
            if f.lower().endswith(".mat"):
                all_files.append(os.path.join(dp, f))
    all_files.sort()
    if not all_files:
        raise FileNotFoundError("No .mat files found under root.")
    logger.info(f"Files scanned: {len(all_files)}")

    # 2) label files
    items = []
    skipped = 0
    for p in all_files:
        lbl_str, lbl = infer_label_from_path(p, cfg.label_mode if cfg.label_mode != "compact8" else "stage10")
        if lbl < 0:
            skipped += 1
            continue
        items.append({"path": p, "label": int(lbl), "label_str": lbl_str})
    logger.info(f"Labeled: {len(items)} | skipped(no label): {skipped}")
    if len(items) == 0:
        raise RuntimeError("No labeled files found.")

    # optional cap files per class (debug)
    if cfg.max_files_per_class and cfg.max_files_per_class > 0:
        byc = {}
        for it in items:
            byc.setdefault(it["label"], []).append(it)
        items2 = []
        for c in sorted(byc):
            items2.extend(byc[c][:cfg.max_files_per_class])
        items = items2
        logger.warning(f"Applied max_files_per_class={cfg.max_files_per_class}. Remaining files={len(items)}")

    # 3) stratified file-wise split (deterministic seed)
    rng_shuffle = np.random.RandomState(cfg.seed)
    by_label = {}
    for it in items:
        by_label.setdefault(it["label"], []).append(it)
    splits = {"train": [], "val": [], "test": []}
    for lbl, arr in by_label.items():
        arr = list(arr)
        rng_shuffle.shuffle(arr)
        n = len(arr)
        n_tr = int(round(cfg.train_ratio * n))
        n_va = int(round(cfg.val_ratio * n))
        # ensure at least one per split if possible
        if n >= 3:
            n_tr = max(1, min(n_tr, n - 2))
            n_va = max(1, min(n_va, n - n_tr - 1))
        tr = arr[:n_tr]
        va = arr[n_tr:n_tr + n_va]
        te = arr[n_tr + n_va:]
        splits["train"].extend(tr)
        splits["val"].extend(va)
        splits["test"].extend(te)

    # shuffle within splits to remove order bias
    for sp in ["train", "val", "test"]:
        rng_shuffle.shuffle(splits[sp])
    logger.info(f"Split file counts: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # 4) count windows per file (with downsample)
    logger.info("Counting windows (exact downsample + length)...")
    counts = {"train": 0, "val": 0, "test": 0}
    kept = {"train": [], "val": [], "test": []}
    for sp in ["train", "val", "test"]:
        for it in splits[sp]:
            nwin = 0
            md = load_mat_robust(it["path"])
            if md is not None:
                cands = recursive_collect_numeric_1d(md)
                sig = choose_3ch_signals(cands)
                if sig is not None:
                    sig_ds = downsample_3ch(sig, cfg.downsample_q, cfg.downsample_mode)
                    L = sig_ds.shape[1]
                    if L >= cfg.length:
                        nwin = (L - cfg.length) // cfg.stride + 1
            if nwin > 0:
                it2 = dict(it)
                it2["n_win"] = int(nwin)
                kept[sp].append(it2)
                counts[sp] += int(nwin)
    logger.info(f"Window counts: {counts}")
    if counts["train"] == 0:
        raise RuntimeError("Zero training windows. Check downsample_q/length/stride.")

    # 5) allocate memmaps
    mms = {}
    for sp in ["train", "val", "test"]:
        N = counts[sp]
        if N <= 0:
            continue
        mms[f"{sp}_sig"] = open_memmap(os.path.join(cfg.out_dir, f"X_{sp}_sig.npy"), (N, 3, cfg.length), dtype="float32")
        mms[f"{sp}_feat"] = open_memmap(os.path.join(cfg.out_dir, f"X_{sp}_feat.npy"), (N, cfg.feat_dim), dtype="float32")
        mms[f"{sp}_y"] = open_memmap(os.path.join(cfg.out_dir, f"y_{sp}.npy"), (N,), dtype="int64")

    # write provisional classes (may be overwritten on compact remap)
    class_map_base = CLASS_MAP_STAGE10 if cfg.label_mode in ("stage10", "compact8") else CLASS_MAP_CLASSIC4
    np.save(os.path.join(cfg.out_dir, "classes.npy"), np.array([class_map_base], dtype=object), allow_pickle=True)

    # 6) TRAIN filter training (train-only)
    filters = [None, None, None]
    if cfg.use_filter:
        logger.info("Building reservoir for filter training (train-only)...")
        reservoir = [[], [], []]
        seen = 0
        for it in kept["train"]:
            md = load_mat_robust(it["path"])
            if md is None:
                continue
            sig = choose_3ch_signals(recursive_collect_numeric_1d(md))
            if sig is None:
                continue
            sig_ds = downsample_3ch(sig, cfg.downsample_q, cfg.downsample_mode)
            wins = extract_windows(sig_ds, cfg.length, cfg.stride)  # (Nw,3,L)
            if wins.shape[0] == 0:
                continue
            take = min(wins.shape[0], 50)
            idx = rng.choice(wins.shape[0], take, replace=False) if wins.shape[0] > take else np.arange(wins.shape[0])
            ws = wins[idx]  # (take,3,L)
            for c in range(3):
                reservoir[c].append(ws[:, c, :])
            seen += take
            if seen >= cfg.filter_reservoir_per_ch:
                break

        for c in range(3):
            data_1ch = np.concatenate(reservoir[c], axis=0).astype(np.float32) if len(reservoir[c]) else np.zeros((0, cfg.length), dtype=np.float32)
            if data_1ch.shape[0] > cfg.filter_reservoir_per_ch:
                idx = rng.choice(len(data_1ch), cfg.filter_reservoir_per_ch, replace=False)
                data_1ch = data_1ch[idx]
            logger.info(f"Training filter ch{c} on N={len(data_1ch)} windows...")
            model_f = LearnedFilter(cfg.length, taps=cfg.filter_taps)
            model_f = train_filter_mse(model_f, data_1ch, cfg.filter_epochs, cfg.filter_lr, cfg.filter_batch, cfg.seed + 17 * c)
            # save state_dict (safer to save state_dict than full model)
            sd_path = os.path.join(cfg.out_dir, f"learned_filter_ch{c}.pt")
            torch.save(model_f.state_dict(), sd_path)
            filters[c] = model_f
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    else:
        logger.info("Skipping learned filter training (use_filter=False).")

    # 7) main processing: apply filters -> extract features -> write memmaps
    extractor = GPUFeatureExtractor(cfg.length, DEVICE).eval()

    # train-only stats
    x_min = np.full((1, 3, 1), np.inf, dtype=np.float32)
    x_max = np.full((1, 3, 1), -np.inf, dtype=np.float32)
    feat_scaler = StandardScaler()
    scaler_fitted = False

    cursor = {"train": 0, "val": 0, "test": 0}

    logger.info("Writing memmaps (filtered signals + engineered features)...")
    for sp in ["train", "val", "test"]:
        if counts[sp] <= 0:
            continue
        logger.info(f"Processing split={sp} | files={len(kept[sp])} | windows={counts[sp]}")
        for it in kept[sp]:
            md = load_mat_robust(it["path"])
            if md is None:
                continue
            sig = choose_3ch_signals(recursive_collect_numeric_1d(md))
            if sig is None:
                continue
            sig_ds = downsample_3ch(sig, cfg.downsample_q, cfg.downsample_mode)
            wins = extract_windows(sig_ds, cfg.length, cfg.stride)  # (Nw,3,L)
            if wins.shape[0] == 0:
                continue

            n_w = wins.shape[0]
            start = 0
            while start < n_w:
                end = min(n_w, start + cfg.write_batch)
                batch = wins[start:end]  # (B,3,L)
                B = batch.shape[0]

                bt = torch.from_numpy(batch).float().to(DEVICE)  # (B,3,L)

                with torch.no_grad():
                    if cfg.use_filter:
                        out = torch.empty_like(bt)
                        for c in range(3):
                            # apply learned filter model (if available) per-channel
                            if filters[c] is not None:
                                # filters[c] expects (B,1,L)
                                inp = bt[:, c:c+1, :]
                                xf = filters[c].to(DEVICE)(inp)
                                out[:, c:c+1, :] = xf
                            else:
                                out[:, c:c+1, :] = bt[:, c:c+1, :]
                        bt_f = out
                    else:
                        bt_f = bt

                    feats = []
                    for c in range(3):
                        feats.append(extractor.extract(bt_f[:, c, :]))  # each (B,7)
                    feats_t = torch.cat(feats, dim=1)  # (B, feat_dim)

                sig_np = bt_f.detach().cpu().numpy().astype(np.float32)
                feat_np = feats_t.detach().cpu().numpy().astype(np.float32)

                # update train-only stats
                if sp == "train":
                    bmin = sig_np.min(axis=(0, 2), keepdims=True)
                    bmax = sig_np.max(axis=(0, 2), keepdims=True)
                    x_min = np.minimum(x_min, bmin)
                    x_max = np.maximum(x_max, bmax)

                    if feat_np.shape[0] > 0:
                        feat_scaler.partial_fit(feat_np)
                        scaler_fitted = True

                cur = cursor[sp]
                cur_end = cur + B
                if cur_end > counts[sp]:
                    B = counts[sp] - cur
                    if B <= 0:
                        break
                    sig_np = sig_np[:B]
                    feat_np = feat_np[:B]
                    cur_end = cur + B

                mms[f"{sp}_sig"][cur:cur_end] = sig_np
                mms[f"{sp}_feat"][cur:cur_end] = feat_np
                mms[f"{sp}_y"][cur:cur_end] = int(it["label"])

                cursor[sp] = cur_end
                start = end

            if cursor[sp] >= counts[sp]:
                break

    if not scaler_fitted:
        raise RuntimeError("Feature scaler could not be fitted (no train features).")

    # 8) finalize scaler & save
    scaler_mean = feat_scaler.mean_.astype(np.float32)
    scaler_scale = feat_scaler.scale_.astype(np.float32)
    np.save(os.path.join(cfg.out_dir, "scaler_mean.npy"), scaler_mean)
    np.save(os.path.join(cfg.out_dir, "scaler_scale.npy"), scaler_scale)
    joblib.dump(feat_scaler, os.path.join(cfg.out_dir, "feature_scaler.joblib"))
    np.savez(os.path.join(cfg.out_dir, "signal_norm_stats.npz"), x_min=x_min, x_max=x_max)
    logger.info("Saved train-only scaler_mean/scaler_scale and signal_norm_stats.")

    # 9) In-place normalization of signals to [-1, 1] using train min/max
    denom = (x_max - x_min).copy()
    denom[denom == 0] = 1e-6
    logger.info("In-place signal normalization to [-1,1] (train-only min/max) ...")

    def norm_sig(mm: np.memmap, N: int):
        bs = cfg.mmap_write_batch
        for i in range(0, N, bs):
            chunk = mm[i:i+bs].astype(np.float32)
            # chunk shape (B,3,L)
            chunk = (2.0 * (chunk - x_min) / denom - 1.0).astype(np.float32)
            mm[i:i+bs] = chunk

    for sp in ["train", "val", "test"]:
        if counts[sp] > 0:
            norm_sig(mms[f"{sp}_sig"], counts[sp])

    logger.info("In-place feature standardization (train-only scaler) ...")
    def norm_feat(mm: np.memmap, N: int):
        bs = max(1024, cfg.mmap_write_batch)
        m = scaler_mean.reshape(1, -1)
        s = scaler_scale.reshape(1, -1)
        s = np.where(s == 0, 1.0, s)
        for i in range(0, N, bs):
            chunk = mm[i:i+bs].astype(np.float32)
            chunk = ((chunk - m) / s).astype(np.float32)
            mm[i:i+bs] = chunk

    for sp in ["train", "val", "test"]:
        if counts[sp] > 0:
            norm_feat(mms[f"{sp}_feat"], counts[sp])

    # 10) Optional noisy test benches generation (normalized domain)
    if cfg.make_noisy_bench and cfg.snr_list:
        logger.info(f"Generating noisy benchmarks for SNR list: {cfg.snr_list}")
        N = counts["test"]
        if N > 0:
            for snr in cfg.snr_list:
                out_sig = open_memmap(os.path.join(cfg.out_dir, f"test_snr_{snr}_sig.npy"), (N, 3, cfg.length), dtype="float32")
                out_feat = open_memmap(os.path.join(cfg.out_dir, f"test_snr_{snr}_feat.npy"), (N, cfg.feat_dim), dtype="float32")
                bs = cfg.mmap_write_batch
                m = scaler_mean.reshape(1, -1)
                s = np.where(cfg is not None and scaler_scale.reshape(1, -1) is not None, scaler_scale.reshape(1, -1), np.ones_like(m))
                for i in range(0, N, bs):
                    chunk = mms["test_sig"][i:i+bs].astype(np.float32)  # already normalized [-1,1]
                    noisy = add_wgn_np(chunk, float(snr), rng)          # keep in normalized domain
                    bt = torch.from_numpy(noisy).float().to(DEVICE)
                    with torch.no_grad():
                        feats = []
                        extractor = extractor  # alias
                        for c in range(3):
                            feats.append(extractor.extract(bt[:, c, :]))
                        feats_t = torch.cat(feats, dim=1)
                    feat_np = feats_t.cpu().numpy().astype(np.float32)
                    # standardize with train scaler
                    feat_np = (feat_np - m) / s
                    out_sig[i:i+noisy.shape[0]] = noisy
                    out_feat[i:i+feat_np.shape[0]] = feat_np
                logger.info(f"Saved benchmark SNR={snr} memmaps.")

    # 11) Optional compact remap (recommended if some classes missing)
    old2new = None
    new_class_map = None
    if cfg.label_mode == "compact8":
        y_train = np.array(mms["train_y"], dtype=np.int64)
        y_val = np.array(mms["val_y"], dtype=np.int64) if counts["val"] > 0 else np.zeros((0,), dtype=np.int64)
        y_test = np.array(mms["test_y"], dtype=np.int64) if counts["test"] > 0 else np.zeros((0,), dtype=np.int64)
        ytr2, yva2, yte2, old2new, new_class_map = remap_compact_labels(y_train, y_val, y_test, CLASS_MAP_STAGE10)
        mms["train_y"][:] = ytr2
        if counts["val"] > 0: mms["val_y"][:] = yva2
        if counts["test"] > 0: mms["test_y"][:] = yte2
        np.save(os.path.join(cfg.out_dir, "classes.npy"), np.array([new_class_map], dtype=object), allow_pickle=True)
        logger.info(f"[COMPACT REMAP] present old classes: {sorted(old2new.keys())}")
        logger.info(f"[COMPACT REMAP] new classes: {new_class_map}")

    # 12) Save meta & manifest
    def bincount_safe(y: np.ndarray, K: int) -> List[int]:
        if len(y) == 0:
            return [0] * K
        return np.bincount(y.astype(np.int64), minlength=K).tolist()

    if cfg.label_mode == "compact8" and new_class_map is not None:
        num_classes_final = len(new_class_map)
        class_map_final = new_class_map
    elif cfg.label_mode == "classic4":
        num_classes_final = 4
        class_map_final = CLASS_MAP_CLASSIC4
    else:
        num_classes_final = 10
        class_map_final = CLASS_MAP_STAGE10

    ytr = np.array(mms["train_y"], dtype=np.int64)
    yva = np.array(mms["val_y"], dtype=np.int64) if counts["val"] > 0 else np.zeros((0,), dtype=np.int64)
    yte = np.array(mms["test_y"], dtype=np.int64) if counts["test"] > 0 else np.zeros((0,), dtype=np.int64)

    meta = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": str(DEVICE),
        "cfg": cfg.__dict__,
        "counts_windows": counts,
        "cursor_final": cursor,
        "label_mode": cfg.label_mode,
        "num_classes": int(num_classes_final),
        "class_map": {str(k): v for k, v in class_map_final.items()},
        "class_counts": {
            "train": bincount_safe(ytr, num_classes_final),
            "val": bincount_safe(yva, num_classes_final),
            "test": bincount_safe(yte, num_classes_final),
        },
        "old2new_class_map": {str(k): int(v) for k, v in old2new.items()} if old2new else None,
    }
    with open(os.path.join(cfg.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    manifest = {
        "created_at": meta["created_at"],
        "label_mode": cfg.label_mode,
        "files": {
            sp: [{"path": it["path"], "label": it["label"], "label_str": it["label_str"], "n_win": it["n_win"]} for it in kept[sp]]
            for sp in ["train", "val", "test"]
        }
    }
    with open(os.path.join(cfg.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(f"Final cursors: {cursor} vs counts={counts}")
    for sp in ["train", "val", "test"]:
        if cursor[sp] != counts[sp]:
            logger.warning(f"[WARN] cursor mismatch on {sp}: cursor={cursor[sp]} count={counts[sp]} (some windows skipped)")

    logger.info("DONE.")
    logger.info(f"Outputs written to: {cfg.out_dir}")

# =========================
# CLI
# =========================
def parse_args() -> CFG:
    p = argparse.ArgumentParser(description="PU Final UltraPro Preprocess (memmap, leak-free, train-only filters, noisy benches).")
    p.add_argument("--root", type=str, required=True, help="PU root with .mat files")
    p.add_argument("--out_dir", type=str, required=True, help="Output folder for processed memmaps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--length", type=int, default=2048)
    p.add_argument("--stride", type=int, default=1024)
    p.add_argument("--downsample_q", type=int, default=6)
    p.add_argument("--downsample_mode", type=str, default="poly", choices=["poly", "slice"])
    p.add_argument("--train_ratio", type=float, default=0.6)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--label_mode", type=str, default="compact8", choices=["compact8", "stage10", "classic4"])
    p.add_argument("--no_filter", action="store_true")
    p.add_argument("--filter_epochs", type=int, default=3)
    p.add_argument("--filter_lr", type=float, default=1e-3)
    p.add_argument("--filter_batch", type=int, default=64)
    p.add_argument("--filter_reservoir_per_ch", type=int, default=20000)
    p.add_argument("--filter_taps", type=int, default=31)
    p.add_argument("--write_batch", type=int, default=256)
    p.add_argument("--mmap_write_batch", type=int, default=4096)
    p.add_argument("--make_noisy_bench", action="store_true")
    p.add_argument("--snr_list", nargs="+", type=int, default=[-10, -8, -6, -4, -2])
    p.add_argument("--max_files_per_class", type=int, default=0)
    a = p.parse_args()

    cfg = CFG(
        root=a.root,
        out_dir=a.out_dir,
        seed=a.seed,
        length=a.length,
        stride=a.stride,
        downsample_q=a.downsample_q,
        downsample_mode=a.downsample_mode,
        train_ratio=a.train_ratio,
        val_ratio=a.val_ratio,
        test_ratio=a.test_ratio,
        label_mode=a.label_mode,
        use_filter=(not a.no_filter),
        filter_epochs=a.filter_epochs,
        filter_lr=a.filter_lr,
        filter_batch=a.filter_batch,
        filter_reservoir_per_ch=a.filter_reservoir_per_ch,
        filter_taps=a.filter_taps,
        write_batch=a.write_batch,
        mmap_write_batch=a.mmap_write_batch,
        make_noisy_bench=a.make_noisy_bench,
        snr_list=a.snr_list,
        max_files_per_class=a.max_files_per_class,
    )
    return cfg

if __name__ == "__main__":
    logger.info(f"Computation Device: {DEVICE}")
    cfg = parse_args()
    main(cfg)
