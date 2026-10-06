# -*- coding: utf-8 -*-
"""
Preprocess_refactored_v3_production.py
Production-ready preprocessing pipeline for CWRU (advanced)
Author: Adapted for Ali Naderi - 2025
This version:
 - Robust filename->class inference
 - Deterministic seeding + optional deterministic cudnn
 - Vectorized WGN and batched augmentations
 - LearnedFilter with stabilized loss (kurtosis + reconstruction)
 - Save filter state_dicts, scaler, signal stats, and raw sample snapshots
 - Compute & save filter sanity report (robust metrics + top amplified bins)
 - Suppress git stderr to avoid 'fatal: Needed a single revision' noise
 - Detailed provenance (train/val/test file maps) and meta with class counts
Usage:
    python Preprocess_refactored_v3_production.py --data_path <path> --out_dir processed_data_robust
"""
from __future__ import annotations
import os
import re
import json
import argparse
import logging
import sys
import gc
import subprocess
import joblib
from typing import List, Tuple, Optional, Dict, Any
from collections import Counter
from datetime import datetime, timezone
import platform

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ------------------- Logger & Config -------------------
logger = logging.getLogger("preprocess_cwru_prod_v3")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)

DEFAULT_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")

# ------------------- Reproducibility -------------------
def set_global_seed(seed: int, deterministic_cudnn: bool = False):
    """
    Set seed broadly for numpy, python random, torch (CPU & CUDA).
    If deterministic_cudnn is True, try to enable deterministic algorithms.
    """
    seed = int(seed)
    import random as _rd
    np.random.seed(seed)
    _rd.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            logger.warning("torch.use_deterministic_algorithms not available; falling back to cudnn flags.")

# ------------------- Utility Helpers -------------------
def flatten_mat_array(arr):
    try:
        a = np.asarray(arr)
        if a.size == 0:
            return np.array([], dtype=float)
        return a.ravel()
    except Exception:
        return np.array([], dtype=float)


def find_channel_keys_from_mat(mat_dict: dict) -> Dict[str, Optional[np.ndarray]]:
    """Finds best candidate arrays for DE, FE, BA channels in a .mat dict.
    Returns dict with keys 'DE','FE','BA' with numpy arrays or None.
    """
    patterns = {
        'DE': [re.compile(p, re.I) for p in [r'(^|_)de($|_)', r'\bdrive\b', r'de_time', r'accel', r'acce?l?x?']],
        'FE': [re.compile(p, re.I) for p in [r'(^|_)fe($|_)', r'\bfan\b', r'fe_time']],
        'BA': [re.compile(p, re.I) for p in [r'(^|_)ba($|_)', r'\bbase\b', r'ba_time']]
    }
    chosen = {'DE': None, 'FE': None, 'BA': None}

    candidates = []
    for k, v in mat_dict.items():
        if not isinstance(k, str) or k.startswith('__'):
            continue
        arr = flatten_mat_array(v)
        if arr.size > 1024:
            candidates.append((k, arr))

    used_keys = set()
    for ch_name, pats in patterns.items():
        for k, arr in candidates:
            if k in used_keys:
                continue
            for p in pats:
                try:
                    if p.search(k):
                        chosen[ch_name] = arr
                        used_keys.add(k)
                        break
                except Exception:
                    continue
            if chosen[ch_name] is not None:
                break

    # fallback: if any channel missing, pick largest remaining array
    if any(v is None for v in chosen.values()):
        remaining = [(k, a) for (k, a) in candidates if k not in used_keys]
        if remaining:
            remaining.sort(key=lambda x: x[1].size, reverse=True)
            for ch_name in chosen:
                if chosen[ch_name] is None and remaining:
                    chosen[ch_name] = remaining.pop(0)[1]

    return chosen

# ==================== 0 Hp Datasets ========================
# def infer_class_name(fname: str) -> str:
#     """Robust class inference from filename.
#     - Tokenizes on non-alnum to avoid false matches (e.g. 'drive' contains 'ir').
#     - Recognizes Ball, Inner, Outer and sizes 007/014/021.
#     """
#     s = os.path.basename(fname).lower()
#     tokens = re.split(r'[^a-z0-9]+', s)

#     # Normal detection
#     if 'normal' in tokens or any(t == '97' for t in tokens) or s.endswith('_97.mat'):
#         return 'Normal'

#     # Type detection (token-based)
#     ftype = None
#     for t in tokens:
#         if re.fullmatch(r'b(?:all)?\d{2,3}', t) or t == 'b' or t == 'ball':
#             ftype = 'Ball'
#             break
#         if t in ('ir', 'inner'):
#             ftype = 'Inner'
#             break
#         if t in ('or', 'outer'):
#             ftype = 'Outer'
#             break

#     # fallback substring checks (word boundaries)
#     if ftype is None:
#         if re.search(r'(?<![a-z])ir(?![a-z])', s):
#             ftype = 'Inner'
#         elif re.search(r'(?<![a-z])or(?![a-z])', s):
#             ftype = 'Outer'
#         elif re.search(r'(?<![a-z])b0', s):
#             ftype = 'Ball'

#     # Size detection: look for 007, 014, 021
#     fsize = None
#     for t in tokens:
#         if re.search(r'007', t):
#             fsize = '007'; break
#         if re.search(r'014', t):
#             fsize = '014'; break
#         if re.search(r'021', t) or t == '21':
#             fsize = '021'; break
#     if fsize is None:
#         m = re.search(r'(007|014|021)', s)
#         if m:
#             fsize = m.group(1)

#     if ftype and fsize:
#         return f"{ftype}_{fsize}"
#     if ftype:
#         return ftype
#     return 'Unknown'



# ================== 1 Hp Datasets ==============================
# def infer_class_name(fname: str) -> str:
#     """
#     Infers class name based on CWRU file IDs for 1HP (12k Fan End/Drive End).
#     Mapping based on standard CWRU documentation for 1HP load.
#     """
#     s = os.path.basename(fname).lower().replace('.mat', '')
    
#     cwru_1hp_map = {
#         '98':  'Normal',
#         '106': 'Inner_007',
#         '119': 'Ball_007',
#         '131': 'Outer_007',  # Centered @6:00
#         '170': 'Inner_014',
#         '186': 'Ball_014',
#         '198': 'Outer_014',  # Centered @6:00
#         '210': 'Inner_021',
#         '223': 'Ball_021',
#         '235': 'Outer_021'   # Centered @6:00
#     }

#     if s in cwru_1hp_map:
#         return cwru_1hp_map[s]
        
#     for code, label in cwru_1hp_map.items():
#         if code in s:
#             return label

#     tokens = re.split(r'[^a-z0-9]+', s)

#     # Normal detection
#     if 'normal' in tokens or '98' in tokens:
#         return 'Normal'

#     # Type detection
#     ftype = None
#     if 'ball' in tokens or 'b' in tokens: ftype = 'Ball'
#     elif 'inner' in tokens or 'ir' in tokens: ftype = 'Inner'
#     elif 'outer' in tokens or 'or' in tokens: ftype = 'Outer'

#     # Size detection
#     fsize = None
#     if '007' in s: fsize = '007'
#     elif '014' in s: fsize = '014'
#     elif '021' in s: fsize = '021'

#     if ftype and fsize:
#         return f"{ftype}_{fsize}"
    
#     return 'Unknown'

# =================== 2 Hp Datasets =========================
# def infer_class_name(fname: str) -> str:
#     """
#     Infers class name based on CWRU file IDs for 0HP, 1HP, 2HP, 3HP.
#     Mapping based on standard CWRU documentation.
#     """
#     base_name = os.path.basename(fname)
#     s = os.path.splitext(base_name)[0].lower() # e.g. "107"
    
#     cwru_map = {
#         # --- Normal ---
#         '97': 'Normal', '98': 'Normal', '99': 'Normal', '100': 'Normal',
        
#         # --- 12k Drive End Faults ---
#         # Inner Race 007
#         '105': 'Inner_007', '106': 'Inner_007', '107': 'Inner_007', '108': 'Inner_007',
#         # Ball 007
#         '118': 'Ball_007', '119': 'Ball_007', '120': 'Ball_007', '121': 'Ball_007',
#         # Outer Race 007 (@6:00)
#         '130': 'Outer_007', '131': 'Outer_007', '132': 'Outer_007', '133': 'Outer_007',
        
#         # Inner Race 014
#         '169': 'Inner_014', '170': 'Inner_014', '171': 'Inner_014', '172': 'Inner_014',
#         # Ball 014
#         '185': 'Ball_014', '186': 'Ball_014', '187': 'Ball_014', '188': 'Ball_014',
#         # Outer Race 014 (@6:00)
#         '197': 'Outer_014', '198': 'Outer_014', '199': 'Outer_014', '200': 'Outer_014',
        
#         # Inner Race 021
#         '209': 'Inner_021', '210': 'Inner_021', '211': 'Inner_021', '212': 'Inner_021',
#         # Ball 021
#         '222': 'Ball_021', '223': 'Ball_021', '224': 'Ball_021', '225': 'Ball_021',
#         # Outer Race 021 (@6:00)
#         '234': 'Outer_021', '235': 'Outer_021', '236': 'Outer_021', '237': 'Outer_021'
#     }

#     # 1. Exact Match
#     if s in cwru_map:
#         return cwru_map[s]
        
#     # 2. Substring Match
#     for code, label in cwru_map.items():
#         if code in s:
#             return label

#     # 3. Fallback (Old Logic)
#     tokens = re.split(r'[^a-z0-9]+', s)
#     if 'normal' in tokens: return 'Normal'
    
#     ftype = None
#     if 'ball' in tokens or 'b' in tokens: ftype = 'Ball'
#     elif 'inner' in tokens or 'ir' in tokens: ftype = 'Inner'
#     elif 'outer' in tokens or 'or' in tokens: ftype = 'Outer'

#     fsize = None
#     if '007' in s: fsize = '007'
#     elif '014' in s: fsize = '014'
#     elif '021' in s: fsize = '021'

#     if ftype and fsize:
#         return f"{ftype}_{fsize}"
    
#     return 'Unknown'

# ========================= 3 Hp ==============================
def infer_class_name(fname: str) -> str:
    """
    Infers class name based on CWRU file IDs for 0HP, 1HP, 2HP, and 3HP.
    Mapping based on standard CWRU documentation (12k Drive End).
    """
    base_name = os.path.basename(fname)
    s = os.path.splitext(base_name)[0].lower() # e.g. "100", "108"
    
    cwru_map = {
        # --- Normal Baseline ---
        '97': 'Normal',   # 0HP
        '98': 'Normal',   # 1HP
        '99': 'Normal',   # 2HP
        '100': 'Normal',  # 3HP
        
        # --- 0.007" Faults ---
        # Inner Race
        '105': 'Inner_007', '106': 'Inner_007', '107': 'Inner_007', '108': 'Inner_007',
        # Ball
        '118': 'Ball_007',  '119': 'Ball_007',  '120': 'Ball_007',  '121': 'Ball_007',
        # Outer Race (@6:00)
        '130': 'Outer_007', '131': 'Outer_007', '132': 'Outer_007', '133': 'Outer_007',
        
        # --- 0.014" Faults ---
        # Inner Race
        '169': 'Inner_014', '170': 'Inner_014', '171': 'Inner_014', '172': 'Inner_014',
        # Ball
        '185': 'Ball_014',  '186': 'Ball_014',  '187': 'Ball_014',  '188': 'Ball_014',
        # Outer Race (@6:00)
        '197': 'Outer_014', '198': 'Outer_014', '199': 'Outer_014', '200': 'Outer_014',
        
        # --- 0.021" Faults ---
        # Inner Race
        '209': 'Inner_021', '210': 'Inner_021', '211': 'Inner_021', '212': 'Inner_021',
        # Ball
        '222': 'Ball_021',  '223': 'Ball_021',  '224': 'Ball_021',  '225': 'Ball_021',
        # Outer Race (@6:00)
        '234': 'Outer_021', '235': 'Outer_021', '236': 'Outer_021', '237': 'Outer_021'
    }

    # 1. Exact Match
    if s in cwru_map:
        return cwru_map[s]
        
    # 2. Substring Match
    for code, label in cwru_map.items():
        if code in s:
            return label

    # 3. Fallback
    tokens = re.split(r'[^a-z0-9]+', s)
    if 'normal' in tokens: return 'Normal'
    
    ftype = None
    if 'ball' in tokens or 'b' in tokens: ftype = 'Ball'
    elif 'inner' in tokens or 'ir' in tokens: ftype = 'Inner'
    elif 'outer' in tokens or 'or' in tokens: ftype = 'Outer'

    fsize = None
    if '007' in s: fsize = '007'
    elif '014' in s: fsize = '014'
    elif '021' in s: fsize = '021'

    if ftype and fsize:
        return f"{ftype}_{fsize}"
    
    return 'Unknown'

# ------------------- Augmentations ------------------- (vectorized)
def wgn_batch(batch: np.ndarray, snrs: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """
    Vectorized WGN for a batch.
    batch: (B, C, L)
    snrs: (B,) in dB (per-sample SNR)
    """
    if batch.ndim != 3:
        raise ValueError("wgn_batch expects batch shape (B, C, L)")
    B, C, L = batch.shape
    out = batch.astype(np.float32)
    sig_pwr = np.mean(out ** 2, axis=2, keepdims=True)  # (B,C,1)
    snr_lin = 10.0 ** (snrs[:, None, None] / 10.0)       # (B,1,1)
    noise_pwr = sig_pwr / (snr_lin + 1e-12)
    noise = rng.randn(B, C, L).astype(np.float32) * np.sqrt(noise_pwr)
    return (out + noise).astype(np.float32)


def apply_augmentations(batch: np.ndarray, rng: np.random.RandomState, params: dict) -> np.ndarray:
    """Apply time-shift and gain augmentations in-place on a batch: (B, C, L)"""
    B, C, L = batch.shape
    # Time Shift per-sample (vectorized-ish)
    p_shift = params.get('time_shift_prob', 0)
    max_shift = int(params.get('time_shift_max', min(50, L // 10)))
    if p_shift > 0:
        shifts_mask = rng.rand(B) < p_shift
        shift_vals = rng.randint(-max_shift, max_shift + 1, size=B)
        for i in np.where(shifts_mask)[0]:
            s = int(shift_vals[i])
            batch[i] = np.roll(batch[i], s, axis=-1)

    # Gain per-sample
    p_gain = params.get('gain_prob', 0)
    low = params.get('gain_low', 0.9)
    high = params.get('gain_high', 1.1)
    if p_gain > 0:
        gains_mask = rng.rand(B) < p_gain
        gains = rng.uniform(low, high, size=B)
        for i in np.where(gains_mask)[0]:
            batch[i] = batch[i] * float(gains[i])
    return batch

# ------------------- Feature Extraction (GPU) -------------------
class GPUFeatureExtractor(torch.nn.Module):
    def __init__(self, fs=12000, length=2048, device=DEVICE):
        super().__init__()
        self.fs = fs
        self.length = length
        self.device = device
        self.nperseg = 256
        self.hop = max(1, self.nperseg // 4)
        self.register_buffer("window_buf", torch.hann_window(self.nperseg), persistent=False)

    def extract(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        x_tensor: (B, L) on device
        returns: (B, feat_dim)
        """
        x = x_tensor.to(self.device)
        window = self.window_buf.to(x.device)
        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-12)
        peak = torch.max(torch.abs(x), dim=1, keepdim=True)[0]
        crest = peak / (rms + 1e-9)
        mean_t = torch.mean(x, dim=1, keepdim=True)
        std_t = torch.std(x, dim=1, keepdim=True) + 1e-9
        skew_t = torch.mean((x - mean_t) ** 3, dim=1, keepdim=True) / (std_t ** 3 + 1e-12)
        kurt = torch.mean((x - mean_t) ** 4, dim=1, keepdim=True) / (std_t ** 4 + 1e-12)
        shape_factor = rms / (torch.mean(torch.abs(x), dim=1, keepdim=True) + 1e-9)

        x_pad = F.pad(x, (0, self.nperseg))
        frames = x_pad.unfold(1, self.nperseg, self.hop)
        win_frames = frames * window
        fft = torch.fft.rfft(win_frames, dim=2)
        mag = torch.abs(fft)
        spec_energy = torch.mean(torch.sum(mag ** 2, dim=2), dim=1, keepdim=True) / (self.length + 1e-12)

        fft_full = torch.fft.fft(win_frames, dim=2)
        h = torch.zeros(self.nperseg, device=x.device)
        h[0] = 1
        h[1:self.nperseg // 2] = 2
        env = torch.abs(torch.fft.ifft(fft_full * h, dim=2))
        env_mean = torch.mean(torch.mean(env, dim=2), dim=1, keepdim=True)
        env_std = torch.mean(torch.std(env, dim=2), dim=1, keepdim=True)

        feats = torch.cat([
            mean_t, std_t, rms, skew_t, kurt, peak,
            crest, shape_factor, spec_energy, env_mean, env_std
        ], dim=1)
        return feats

# ------------------- Learned Filter -------------------
class LearnedFilter(nn.Module):
    def __init__(self, length=2048):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 7, padding=3, bias=False)
        self.act = nn.PReLU()
        self.n_freq = length // 2 + 1
        self.gate = nn.Linear(self.n_freq, self.n_freq)
        nn.init.dirac_(self.conv.weight)
        nn.init.constant_(self.gate.weight, 0.0)
        nn.init.constant_(self.gate.bias, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        xt = self.act(self.conv(x)).squeeze(1)
        X = torch.fft.rfft(xt)
        mag = torch.abs(X)
        g = torch.sigmoid(self.gate(mag))
        g = torch.clamp(g, min=0.01, max=0.99)
        return torch.fft.irfft(X * g, n=x.shape[-1])

def train_filter(model: LearnedFilter, data_np: np.ndarray, epochs: int = 3, batch_size: int = 64,
                 lr: float = 1e-3, rng: Optional[np.random.RandomState] = None, seed: Optional[int] = None,
                 recon_alpha: float = 0.5, kurt_alpha: float = 0.5):
    """
    Train LearnedFilter on 2D numpy signals (N, L).
    Loss = -kurt_alpha * mean(kurtosis) + recon_alpha * MSE(out, input)
    """
    model.to(DEVICE)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    if len(data_np) == 0:
        logger.warning("train_filter called with empty data. returning untrained model.")
        model.eval()
        return model

    if len(data_np) > 2000:
        if rng is None:
            indices = np.random.choice(len(data_np), 2000, replace=False)
        else:
            indices = rng.choice(len(data_np), 2000, replace=False)
        data_sub = data_np[indices]
    else:
        data_sub = data_np

    data_t = torch.from_numpy(data_sub).float().to(DEVICE)
    ds = TensorDataset(data_t)

    if seed is None:
        g = None
    else:
        g = torch.Generator()
        g.manual_seed(int(seed))

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)

    for ep in range(epochs):
        epoch_loss = 0.0
        n_seen = 0
        for (xb,) in dl:
            opt.zero_grad()
            out = model(xb)
            mean = out.mean(dim=1, keepdim=True)
            diff = out - mean
            m4 = torch.mean(diff ** 4, dim=1)
            m2 = torch.mean(diff ** 2, dim=1)
            kurt = m4 / (m2 ** 2 + 1e-6)
            recon_loss = torch.mean((out - xb) ** 2)
            loss = -kurt_alpha * torch.mean(kurt) + recon_alpha * recon_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            epoch_loss += float(loss.item()) * xb.size(0)
            n_seen += xb.size(0)
        if n_seen > 0:
            logger.info(f"[train_filter] ep {ep+1}/{epochs} loss={epoch_loss / n_seen:.6f}")
        else:
            logger.info(f"[train_filter] ep {ep+1}/{epochs} no samples seen")
    model.eval()
    return model

# ------------------- Core helpers -------------------
def process_file_time_split(de: np.ndarray, fe: np.ndarray, ba: np.ndarray, length: int, samples_per_file: int) -> List[np.ndarray]:
    total_len = int(de.size)
    if total_len < length:
        return []

    max_possible = max(1, total_len - length + 1)
    if samples_per_file >= max_possible:
        starts = np.arange(0, total_len - length + 1, dtype=int)
    else:
        starts = np.linspace(0, total_len - length, num=samples_per_file, dtype=int)

    samples = []
    for s in starts:
        s = int(s)
        samples.append(np.stack([de[s:s + length], fe[s:s + length], ba[s:s + length]]).astype(np.float32))
    return samples

def safe_stack(X_list: List[np.ndarray], name: str, length: int) -> np.ndarray:
    if len(X_list) == 0:
        logger.warning(f"No samples for {name}. Returning empty array.")
        return np.zeros((0, 3, length), dtype=np.float32)
    return np.stack(X_list, axis=0)

# ------------------- Sanity utilities for filters -------------------
def robust_amplification_metrics(spec_before: np.ndarray, spec_after: np.ndarray, thresh: float = 1e-3) -> Dict[str, Any]:
    """Compute robust metrics ignoring tiny bins."""
    mask = spec_before > thresh
    if mask.sum() == 0:
        return {"mean": None, "median": None, "p95": None, "max": None, "n_masked": 0}
    ratios = (spec_after[mask] + 1e-12) / (spec_before[mask] + 1e-12)
    return {
        "mean": float(np.mean(ratios)),
        "median": float(np.median(ratios)),
        "p95": float(np.percentile(ratios, 95)),
        "max": float(np.max(ratios)),
        "n_masked": int(mask.sum())
    }

def top_amplified_bins(spec_before: np.ndarray, spec_after: np.ndarray, thresh: float = 1e-3, k: int = 10):
    mask = spec_before > thresh
    if mask.sum() == 0:
        return []
    ratios = (spec_after + 1e-12) / (spec_before + 1e-12)
    idxs = np.where(mask)[0]
    r_masked = ratios[mask]
    order = np.argsort(-r_masked)
    top = [{"bin": int(idxs[i]), "ratio": float(r_masked[i])} for i in order[:k]]
    return top

# ------------------- Main pipeline -------------------
def prepro_final(data_path: str, out_dir_name: str, length: int = 2048, target_train_total: int = 10000,
                 snr_list: Optional[List[int]] = None, seed: int = DEFAULT_SEED,
                 deterministic_cudnn: bool = False, save_filters: bool = True,
                 raw_sample_count: int = 32, filter_train_params: dict = None) -> Tuple[str, Dict[str, Any]]:
    """
    Main preprocessing pipeline.
    Returns: (out_npz, meta)
    """
    if snr_list is None:
        snr_list = [-10, -8, -6, -4, -2]

    if filter_train_params is None:
        filter_train_params = {"epochs": 3, "lr": 1e-3, "batch_size": 64, "recon_alpha": 0.5, "kurt_alpha": 0.5}

    set_global_seed(seed, deterministic_cudnn)
    rng = np.random.RandomState(seed)

    files = [f for f in os.listdir(data_path) if f.lower().endswith('.mat')]
    files.sort()
    if len(files) == 0:
        raise RuntimeError(f"No .mat files found in {data_path}")

    labels = [infer_class_name(f) for f in files]
    unique_labels = sorted(list(set(labels)))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    class_counts = Counter(labels)

    logger.info(f"Found {len(files)} files across {len(unique_labels)} classes: {unique_labels}")

    per_class_target = max(200, int(np.ceil(target_train_total / max(1, len(unique_labels)))))
    logger.info(f"Target per-class (approx) for train: {per_class_target}")

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []

    train_file_map = []
    val_file_map = []
    test_file_map = []
    train_counter = 0
    val_counter = 0
    test_counter = 0

    logger.info(f"Scanning {len(files)} files...")
    for fname, label_str in zip(tqdm(files), labels):
        full = os.path.join(data_path, fname)
        try:
            mat = loadmat(full)
        except Exception as e:
            logger.warning(f"Failed to load {fname}: {e}")
            continue

        chans = find_channel_keys_from_mat(mat)
        if chans['DE'] is None:
            logger.warning(f"File {fname} has no DE channel -> skipped")
            continue

        de = chans['DE'].astype(np.float32)
        fe = chans['FE'].astype(np.float32) if chans['FE'] is not None else np.zeros_like(de)
        ba = chans['BA'].astype(np.float32) if chans['BA'] is not None else np.zeros_like(de)

        n = de.size
        i1, i2 = int(0.7 * n), int(0.85 * n)
        n_files = class_counts[label_str]
        tgt_per_file = max(1, int(np.ceil(per_class_target / max(1, n_files))))

        tr = process_file_time_split(de[:i1], fe[:i1], ba[:i1], length, tgt_per_file)
        va = process_file_time_split(de[i1:i2], fe[i1:i2], ba[i1:i2], length, max(1, tgt_per_file // 4))
        te = process_file_time_split(de[i2:], fe[i2:], ba[i2:], length, max(1, tgt_per_file // 4))

        if len(tr) > 0:
            start = train_counter
            train_counter += len(tr)
            train_file_map.append({"file": fname, "label": label_str, "start": start, "end": train_counter, "n_samples": len(tr)})
        if len(va) > 0:
            start = val_counter
            val_counter += len(va)
            val_file_map.append({"file": fname, "label": label_str, "start": start, "end": val_counter, "n_samples": len(va)})
        if len(te) > 0:
            start = test_counter
            test_counter += len(te)
            test_file_map.append({"file": fname, "label": label_str, "start": start, "end": test_counter, "n_samples": len(te)})

        X_train_list.extend(tr); y_train_list.extend([label_to_idx[label_str]] * len(tr))
        X_val_list.extend(va); y_val_list.extend([label_to_idx[label_str]] * len(va))
        X_test_list.extend(te); y_test_list.extend([label_to_idx[label_str]] * len(te))

        logger.debug(f"File {fname} -> tr:{len(tr)} va:{len(va)} te:{len(te)} tgt_per_file:{tgt_per_file}")

    X_train = safe_stack(X_train_list, 'X_train', length)
    X_val = safe_stack(X_val_list, 'X_val', length)
    X_test = safe_stack(X_test_list, 'X_test', length)

    y_train = np.array(y_train_list, dtype=np.int64)
    y_val = np.array(y_val_list, dtype=np.int64)
    y_test = np.array(y_test_list, dtype=np.int64)

    logger.info(f"Raw Stats -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    if len(X_train) == 0:
        raise RuntimeError("No training samples collected. Aborting.")

    # class counts diagnostics
    counts_train = dict(Counter(y_train.tolist()))
    counts_val = dict(Counter(y_val.tolist()))
    counts_test = dict(Counter(y_test.tolist()))
    logger.info(f"Per-class train counts (sample): {counts_train}")

    # Save raw sample snapshot BEFORE filtering for later sanity-checks
    snapshot_dir = os.path.join(data_path, out_dir_name)
    os.makedirs(snapshot_dir, exist_ok=True)
    raw_snapshot_path = os.path.join(snapshot_dir, "raw_snapshot.npz")
    try:
        sample_cnt = min(raw_sample_count, len(X_test))
        np.savez_compressed(raw_snapshot_path,
                            X_train_raw=X_train[:sample_cnt],
                            X_val_raw=X_val[:sample_cnt],
                            X_test_raw=X_test[:sample_cnt])
        logger.info(f"Saved raw snapshot (first {sample_cnt} samples) to {raw_snapshot_path}")
    except Exception as e:
        logger.warning(f"Failed to save raw snapshot: {e}")

    # ---------------- Train Learned Filters (only on train signals) ----------------
    logger.info("Training Learned Filters on train signals only...")
    filters = []
    for c in range(3):
        lf = LearnedFilter(length)
        if X_train.shape[0] == 0:
            subset = np.array([], dtype=np.float32)
        else:
            all_ch = X_train[:, c, :]
            n_samples = min(2000, len(all_ch))
            if n_samples == 0:
                subset = np.array([], dtype=np.float32)
            else:
                indices = rng.choice(len(all_ch), n_samples, replace=False)
                subset = all_ch[indices]
        lf = train_filter(lf, subset, epochs=int(filter_train_params.get("epochs", 3)),
                          lr=float(filter_train_params.get("lr", 1e-3)),
                          batch_size=int(filter_train_params.get("batch_size", 64)),
                          rng=rng, seed=seed,
                          recon_alpha=float(filter_train_params.get("recon_alpha", 0.5)),
                          kurt_alpha=float(filter_train_params.get("kurt_alpha", 0.5)))
        lf.eval()
        filters.append(lf)

        logger.info(f"Filtering Channel {c} (applying learned filter) ...")
        with torch.no_grad():
            for dataset in [X_train, X_val, X_test]:
                N = len(dataset)
                if N == 0:
                    continue
                for i in range(0, N, 128):
                    batch_np = dataset[i:i + 128, c, :].astype(np.float32)
                    batch_t = torch.from_numpy(batch_np).float().to(DEVICE)
                    out = lf(batch_t)
                    out_np = out.cpu().numpy()
                    dataset[i:i + out_np.shape[0], c, :] = out_np
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        gc.collect()

    # ---------------- Feature Extraction & Augmentation ----------------
    extractor = GPUFeatureExtractor(length=length, device=DEVICE)
    extractor.eval()

    aug_params = {
        'time_shift_prob': 0.5,
        'time_shift_max': 50,
        'gain_prob': 0.5,
        'gain_low': 0.9,
        'gain_high': 1.1
    }

    def process_batch_fn(X: np.ndarray, train: bool = False, snr: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        out_x = []
        out_f = []
        bs = 128
        for i in range(0, len(X), bs):
            batch = X[i:i + bs].copy()
            if batch.size == 0:
                continue
            if train:
                apply_augmentations(batch, rng, aug_params)
                snrs = rng.uniform(-8, 15, size=len(batch)).astype(np.float32)
                batch = wgn_batch(batch, snrs, rng)
            elif snr is not None:
                snrs = np.full(len(batch), snr, dtype=np.float32)
                batch = wgn_batch(batch, snrs, rng)

            feats_per_chan = []
            with torch.no_grad():
                for c in range(3):
                    t_sig = torch.from_numpy(batch[:, c, :]).float().to(DEVICE)
                    f = extractor.extract(t_sig).cpu().numpy()
                    feats_per_chan.append(f)
            feats = np.concatenate(feats_per_chan, axis=1).astype(np.float32)

            out_x.append(batch.astype(np.float32))
            out_f.append(feats)

        if len(out_x) == 0:
            feat_dim = 11 * 3
            return np.zeros((0, 3, length), dtype=np.float32), np.zeros((0, feat_dim), dtype=np.float32)
        return np.concatenate(out_x, axis=0), np.concatenate(out_f, axis=0)

    logger.info("Processing Train (augmented)...")
    Xt, Et = process_batch_fn(X_train, train=True)

    logger.info("Processing Val (0 dB)...")
    Xv, Ev = process_batch_fn(X_val, train=False, snr=0)

    logger.info("Processing Test (clean)...")
    Xte, Ete = process_batch_fn(X_test, train=False, snr=None)

    # ---------------- Normalization & Scaler ----------------
    logger.info("Normalizing signals and features...")

    if Xt.size > 0:
        x_min = Xt.min(axis=(0, 2), keepdims=True).astype(np.float32)
        x_max = Xt.max(axis=(0, 2), keepdims=True).astype(np.float32)
    else:
        x_min = np.zeros((1, 3, 1), dtype=np.float32)
        x_max = np.ones((1, 3, 1), dtype=np.float32)

    def norm_signal(x: np.ndarray) -> np.ndarray:
        denom = (x_max - x_min).copy()
        denom[denom == 0] = 1e-6
        return (2.0 * (x - x_min) / denom - 1.0).astype(np.float32)

    if Et is None:
        Et = np.zeros((0, 0), dtype=np.float32)
    if Et.ndim == 1:
        Et = Et.reshape(-1, 1)

    if Et.shape[0] > 0 and Et.ndim == 2:
        scaler = StandardScaler().fit(Et)
        Et_s = scaler.transform(Et).astype(np.float32)
        Ev_s = scaler.transform(Ev).astype(np.float32) if Ev.shape[0] > 0 else np.zeros((0, Et.shape[1]), dtype=np.float32)
        Ete_s = scaler.transform(Ete).astype(np.float32) if Ete.shape[0] > 0 else np.zeros((0, Et.shape[1]), dtype=np.float32)
        feat_dim = Et.shape[1]
    else:
        logger.warning("No training features to fit scaler. Using identity / safe empty arrays.")
        scaler = StandardScaler()
        feat_dim = Ev.shape[1] if (Ev.ndim == 2 and Ev.shape[1] > 0) else (Ete.shape[1] if (Ete.ndim == 2 and Ete.shape[1] > 0) else 33)
        Et_s = np.zeros((0, feat_dim), dtype=np.float32)
        Ev_s = Ev.astype(np.float32) if Ev.ndim == 2 else np.zeros((0, feat_dim), dtype=np.float32)
        Ete_s = Ete.astype(np.float32) if Ete.ndim == 2 else np.zeros((0, feat_dim), dtype=np.float32)

    # ---------------- Shuffle train set deterministically ----------------
    n_train = len(y_train)
    if n_train != Xt.shape[0]:
        logger.warning("y_train length and X_train length mismatch before shuffle; reconciling by using X_train length as ground truth.")
        n_train = Xt.shape[0]
        if y_train.shape[0] != n_train:
            if y_train.shape[0] >= n_train:
                y_train = y_train[:n_train]
            else:
                y_train = np.pad(y_train, (0, n_train - y_train.shape[0]), mode='edge')

    perm = rng.permutation(n_train)
    Xt = Xt[perm]
    if Et_s.shape[0] == n_train:
        Et_s = Et_s[perm]
    y_train = y_train[perm]

    try:
        assert Xt.shape[0] == Et_s.shape[0] == y_train.shape[0]
    except AssertionError:
        logger.error("Dimension mismatch after processing: Xt, Et_s, y_train shapes: %s, %s, %s", Xt.shape, Et_s.shape, y_train.shape)
        nmin = min(Xt.shape[0], Et_s.shape[0], y_train.shape[0])
        Xt = Xt[:nmin]
        Et_s = Et_s[:nmin]
        y_train = y_train[:nmin]
        logger.info("Reconciled to minimum length %d", nmin)

    # ---------------- Save processed arrays ----------------
    save_dict = {
        'X_train_sig': norm_signal(Xt), 'X_train_feat': Et_s, 'y_train': y_train,
        'X_val_sig': norm_signal(Xv) if Xv.size > 0 else np.zeros((0, 3, length), dtype=np.float32),
        'X_val_feat': Ev_s,   'y_val': y_val,
        'X_test_sig': norm_signal(Xte) if Xte.size > 0 else np.zeros((0, 3, length), dtype=np.float32),
        'X_test_feat': Ete_s, 'y_test': y_test
    }

    logger.info("Generating benchmarks for SNR list: %s", snr_list)
    for snr in snr_list:
        xn, en = process_batch_fn(X_test, train=False, snr=snr)
        save_dict[f'test_snr_{snr}_sig'] = norm_signal(xn) if xn.size > 0 else np.zeros((0, 3, length), dtype=np.float32)
        if en.shape[0] > 0:
            try:
                save_dict[f'test_snr_{snr}_feat'] = scaler.transform(en).astype(np.float32) if (hasattr(scaler, "mean_") and en.shape[0] > 0) else en.astype(np.float32)
            except Exception:
                save_dict[f'test_snr_{snr}_feat'] = en.astype(np.float32)
        else:
            save_dict[f'test_snr_{snr}_feat'] = np.zeros((0, feat_dim), dtype=np.float32)

    out_path = os.path.join(data_path, out_dir_name)
    os.makedirs(out_path, exist_ok=True)
    out_npz = os.path.join(out_path, "processed_data_robust_v3.npz")

    # --- SAVE EXTRA ARTIFACTS FOR REPRODUCIBILITY ---
    scaler_path = None
    signal_stats_path = None
    try:
        scaler_path = os.path.join(out_path, "feature_scaler.joblib")
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved feature scaler: {scaler_path}")
    except Exception as e:
        logger.warning(f"Failed to save scaler: {e}")

    try:
        signal_stats_path = os.path.join(out_path, "signal_norm_stats.npz")
        np.savez_compressed(signal_stats_path, x_min=x_min, x_max=x_max)
        logger.info(f"Saved signal normalization stats: {signal_stats_path}")
    except Exception as e:
        logger.warning(f"Failed to save signal norm stats: {e}")

    learned_filter_files = []
    if save_filters:
        try:
            for i, lf in enumerate(filters):
                fpath = os.path.join(out_path, f"learned_filter_ch{i}.pt")
                # safer save: move weights to CPU before saving (avoids CUDA tensors in file)
                try:
                    state_cpu = {k: v.cpu() for k, v in lf.state_dict().items()}
                    torch.save(state_cpu, fpath)
                    learned_filter_files.append(os.path.basename(fpath))
                except Exception as e:
                    logger.warning(f"Failed to save learned filter ch{i}: {e}")
                learned_filter_files.append(os.path.basename(fpath))
            logger.info("Saved learned filter state_dicts.")
        except Exception as e:
            logger.warning(f"Failed to save learned filters: {e}")

    # ---------------- Filter sanity report (apply filters to raw snapshot and compute robust metrics) ----------------
    filter_sanity = {}
    try:
        rs = np.load(raw_snapshot_path)
        X_test_raw = rs.get('X_test_raw', None)
        if X_test_raw is None:
            logger.warning("No raw X_test snapshot found for filter sanity check.")
        else:
            sample_cnt = X_test_raw.shape[0]
            for ch, lf in enumerate(filters):
                lf.eval()
                # determine model device (works if model on cpu or cuda)
                try:
                    model_device = next(lf.parameters()).device
                except StopIteration:
                    model_device = torch.device("cpu")

                with torch.no_grad():
                    inp = torch.from_numpy(X_test_raw[:, ch, :].astype(np.float32))
                    # move input to model device
                    inp = inp.to(model_device)
                    out_t = lf(inp)  # output on same device
                    out = out_t.cpu().numpy()  # bring back to CPU for numpy ops

                spec_before = np.mean(np.abs(np.fft.rfft(X_test_raw[:, ch, :], axis=1)), axis=0)
                spec_after  = np.mean(np.abs(np.fft.rfft(out, axis=1)), axis=0)
                metrics = robust_amplification_metrics(spec_before, spec_after, thresh=1e-3)
                top_bins = top_amplified_bins(spec_before, spec_after, thresh=1e-3, k=10)
                filter_sanity[f"ch{ch}"] = {"metrics": metrics, "top_bins": top_bins}
            # save report
            try:
                sanity_path = os.path.join(out_path, "filter_sanity.json")
                with open(sanity_path, 'w', encoding='utf8') as fh:
                    json.dump(filter_sanity, fh, indent=2)
                logger.info(f"Saved filter sanity report: {sanity_path}")
            except Exception as e:
                logger.warning(f"Failed to save filter sanity report: {e}")
    except Exception as e:
        logger.warning(f"Failed to compute filter sanity report: {e}")


    # meta construction
    meta = {
        "class_map": label_to_idx,
        "idx_to_label": {v: k for k, v in label_to_idx.items()},
        "snr_list": snr_list,
        "length": length,
        "feat_dim": int(feat_dim),
        "n_train": int(len(Xt)),
        "n_val": int(len(Xv)),
        "n_test": int(len(Xte)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_at_local": datetime.now().isoformat(),
        "class_counts": {
            "train": counts_train,
            "val": counts_val,
            "test": counts_test
        }
    }

    # provenance maps
    if train_file_map:
        meta["train_file_map"] = train_file_map
    if val_file_map:
        meta["val_file_map"] = val_file_map
    if test_file_map:
        meta["test_file_map"] = test_file_map

    # runtime / reproducibility info
    try:
        git_cwd = os.path.dirname(__file__) or os.getcwd()
        if os.path.exists(os.path.join(git_cwd, ".git")):
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=git_cwd,
                stderr=subprocess.DEVNULL
            ).decode().strip()
        else:
            git_hash = None
    except Exception:
        git_hash = None
    meta.update({
        "scaler_file": os.path.basename(scaler_path) if scaler_path is not None else None,
        "signal_norm_file": os.path.basename(signal_stats_path) if signal_stats_path is not None else None,
        "learned_filter_files": learned_filter_files if learned_filter_files else None,
        "filter_sanity_file": os.path.basename(sanity_path) if 'sanity_path' in locals() else None,
        "git_hash": git_hash,
        "seed": int(seed),
        "python_version": sys.version.splitlines()[0],
        "platform": platform.platform()
    })

    logger.info(f"Saving processed data to {out_npz} ...")
    try:
        np.savez_compressed(out_npz, **save_dict)
        logger.info("Processed NPZ saved.")
    except Exception as e:
        logger.error(f"Failed to save processed npz: {e}")
        raise

    try:
        with open(os.path.join(out_path, "meta.json"), 'w', encoding='utf8') as f:
            json.dump(meta, f, indent=2)
        logger.info("Meta written.")
    except Exception as e:
        logger.warning(f"Failed to save meta.json: {e}")

    logger.info("Done!")
    return out_npz, meta

# ------------------- CLI -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\3HP", help='Path to folder with .mat files')
    parser.add_argument('--out_dir', default='processed_data_robust_3hp', help='Output subfolder name (created under data_path)')
    parser.add_argument('--length', type=int, default=2048)
    parser.add_argument('--target_train_total', type=int, default=10000)
    parser.add_argument('--snr_list', nargs='+', type=int, default=[-10, -8, -6, -4, -2])
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--deterministic', action='store_true', help='Set cuDNN deterministic flags (may reduce perf)')
    parser.add_argument('--no_save_filters', action='store_true', help='Do not save learned filter state_dicts')
    parser.add_argument('--raw_sample_count', type=int, default=32, help='How many raw samples to snapshot for sanity checks')
    parser.add_argument('--filter_epochs', type=int, default=3, help='Epochs to train learned filters')
    parser.add_argument('--filter_lr', type=float, default=1e-3, help='Learning rate for learned filters')
    parser.add_argument('--filter_batch_size', type=int, default=64, help='Batch size for training filters')
    parser.add_argument('--filter_recon_alpha', type=float, default=0.5, help='Reconstruction weight for filter loss')
    parser.add_argument('--filter_kurt_alpha', type=float, default=0.5, help='Kurtosis weight for filter loss')
    args = parser.parse_args()

    filter_params = {
        "epochs": args.filter_epochs,
        "lr": args.filter_lr,
        "batch_size": args.filter_batch_size,
        "recon_alpha": args.filter_recon_alpha,
        "kurt_alpha": args.filter_kurt_alpha
    }

    prepro_final(args.data_path, args.out_dir, length=args.length, target_train_total=args.target_train_total,
                 snr_list=args.snr_list, seed=args.seed, deterministic_cudnn=args.deterministic,
                 save_filters=(not args.no_save_filters), raw_sample_count=args.raw_sample_count,
                 filter_train_params=filter_params)








import json, os
p = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\3HP\processed_data_robust_3hp\filter_sanity.json"
print("exists:", os.path.exists(p))
print(json.load(open(p, 'r', encoding='utf8')))

import numpy as np, matplotlib.pyplot as plt, torch, glob, os
out_dir = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\3HP\processed_data_robust_3hp"
rs = np.load(os.path.join(out_dir, "raw_snapshot.npz"))
x_test_raw = rs['X_test_raw']  # (N, 3, L)
L = x_test_raw.shape[2]
# load filter (CPU-safe)
state = torch.load(os.path.join(out_dir, "learned_filter_ch0.pt"), map_location='cpu')
# reconstruct minimal model (same arch)
from Preprocess import LearnedFilter
lf = LearnedFilter(length=L)
lf.load_state_dict(state)
lf.eval()
# pick sample index
i = 0
sig = x_test_raw[i, 0, :].astype(np.float32)
with torch.no_grad():
    out = lf(torch.from_numpy(sig).unsqueeze(0)).numpy()[0]
spec_b = np.abs(np.fft.rfft(sig))
spec_a = np.abs(np.fft.rfft(out))
freqs = np.fft.rfftfreq(L, d=1/12000)
plt.figure(figsize=(8,4))
plt.semilogy(freqs, spec_b, label='before')
plt.semilogy(freqs, spec_a, label='after')
plt.xlim(0, 6000)  # up to Nyquist
plt.legend(); plt.show()


import json, os
meta = json.load(open(r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\3HP\processed_data_robust_3hp\meta.json"))
print("class_map:", meta['class_map'])
print("class_counts (train):", meta['class_counts']['train'])
print("train_file_map:", meta.get('train_file_map')[:])


