# -*- coding: utf-8 -*-
"""
Preprocess_IMS_Pro_FINAL.py
=============================================================================
Leakage-Safe, Production-Grade Preprocessing for the IMS Bearing Dataset (3 Sets)
Author: Ali Naderi - 2026-01-01

Fixes in this FINAL version
---------------------------
Proper support for --ims_dirs (explicit dataset folders)
Safe path resolution (never passes None/empty to os.path.join incorrectly)
Clean CLI (no duplicated imports, no global reassignment issues)
Robust handling when some sets are missing
"""

from __future__ import annotations

import os
import sys
import gc
import json
import glob
import argparse
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pathlib import Path

# ------------------- Logger Setup -------------------
logger = logging.getLogger("IMS_Preprocess_Pro_FINAL")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)

# ------------------- Device -------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------- Repro -------------------
def set_global_seed(seed: int, deterministic_cudnn: bool = True) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ------------------- IMS Config -------------------
IMS_CONFIG: Dict[str, Dict[str, Any]] = {
    "Set1": {
        "folder": "1st_test",
        "channels": 8,
        "healthy_files_ratio": 0.20,
        "fault_files_ratio": 0.20,
    },
    "Set2": {
        "folder": "2nd_test",
        "channels": 4,
        "healthy_files_ratio": 0.10,
        "fault_start_ratio": 0.70,
    },
    "Set3": {
        "folder": "4th_test",
        "channels": 4,
        "outer_fault_last_files": 500
    }
}

CLASS_NAMES = {0: "Healthy", 1: "Outer_Race", 2: "Inner_Race", 3: "Roller_Elem"}


# =============================================================================
# Feature extractor (GPU) - 11 features per channel => 33 total (3ch)
# =============================================================================
class GPUFeatureExtractor(nn.Module):
    """Computes 11 statistical features on GPU for 1D signals."""
    def __init__(self, length: int = 2048, device: str = DEVICE):
        super().__init__()
        self.length = length
        self.device = device
        self.nperseg = 256
        self.hop = max(1, self.nperseg // 4)
        self.register_buffer("window_buf", torch.hann_window(self.nperseg), persistent=False)

    @torch.no_grad()
    def extract(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        x_tensor: (B, L)
        returns:  (B, 11)
        """
        x = x_tensor.to(self.device)
        window = self.window_buf.to(x.device)

        mean_t = torch.mean(x, dim=1, keepdim=True)
        std_t = torch.std(x, dim=1, keepdim=True) + 1e-9
        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-12)
        peak = torch.max(torch.abs(x), dim=1, keepdim=True)[0]
        skew = torch.mean((x - mean_t) ** 3, dim=1, keepdim=True) / (std_t ** 3 + 1e-12)
        kurt = torch.mean((x - mean_t) ** 4, dim=1, keepdim=True) / (std_t ** 4 + 1e-12)
        crest = peak / (rms + 1e-9)
        shape = rms / (torch.mean(torch.abs(x), dim=1, keepdim=True) + 1e-9)

        x_pad = F.pad(x, (0, self.nperseg))
        frames = x_pad.unfold(1, self.nperseg, self.hop)
        win_frames = frames * window
        fft = torch.fft.rfft(win_frames, dim=2)
        mag = torch.abs(fft)
        spec_energy = torch.mean(torch.sum(mag ** 2, dim=2), dim=1, keepdim=True) / (self.length + 1e-12)

        fft_full = torch.fft.fft(win_frames, dim=2)
        h = torch.zeros(self.nperseg, device=x.device)
        h[0] = 1.0
        h[1:self.nperseg // 2] = 2.0
        env = torch.abs(torch.fft.ifft(fft_full * h, dim=2))
        env_mean = torch.mean(torch.mean(env, dim=2), dim=1, keepdim=True)
        env_std = torch.mean(torch.std(env, dim=2), dim=1, keepdim=True)

        return torch.cat([mean_t, std_t, rms, skew, kurt, peak, crest, shape, spec_energy, env_mean, env_std], dim=1)


# =============================================================================
# Physics-Informed learned denoising filter (trained ONLY on train split)
# =============================================================================
class LearnedFilter(nn.Module):
    """Lightweight learnable filter: conv + frequency gate."""
    def __init__(self, length: int = 2048):
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
        g = torch.sigmoid(self.gate(torch.abs(X)))
        g = torch.clamp(g, min=0.01, max=0.99)
        return torch.fft.irfft(X * g, n=x.shape[-1])


def train_filter(model: LearnedFilter,
                 data_np: np.ndarray,
                 epochs: int = 3,
                 batch_size: int = 128,
                 lr: float = 1e-3,
                 seed: int = 42) -> LearnedFilter:
    set_global_seed(seed, deterministic_cudnn=True)
    model = model.to(DEVICE).train()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    N = len(data_np)
    if N > 20000:
        idx = np.random.RandomState(seed).choice(N, 20000, replace=False)
        data_np = data_np[idx]
        N = len(data_np)

    for ep in range(epochs):
        perm = np.random.RandomState(seed + ep).permutation(N)
        ep_loss = 0.0
        for i in range(0, N, batch_size):
            batch = data_np[perm[i:i + batch_size]]
            x = torch.from_numpy(batch).float().to(DEVICE)

            noise = torch.randn_like(x) * 0.05
            x_noisy = x + noise

            y_hat = model(x_noisy)
            loss = loss_fn(y_hat, x)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * x.shape[0]

        logger.info(f"  Filter epoch {ep+1}/{epochs} | loss={ep_loss/max(N,1):.6f}")

    model.eval()
    return model


# =============================================================================
# Augmentations
# =============================================================================
def apply_augmentations_and_wgn(batch: np.ndarray,
                               rng: np.random.RandomState,
                               snr_range: Tuple[float, float] = (-12.0, 10.0)) -> np.ndarray:
    B, C, L = batch.shape

    if rng.rand() < 0.5:
        shifts = rng.randint(-int(0.05 * L), int(0.05 * L) + 1, size=B)
        for i in range(B):
            batch[i] = np.roll(batch[i], shifts[i], axis=1)

    if rng.rand() < 0.5:
        gains = rng.uniform(0.9, 1.1, size=B).astype(np.float32)
        batch = batch * gains[:, None, None]

    sig_pwr = np.mean(batch ** 2, axis=2, keepdims=True) + 1e-12
    snrs = rng.uniform(snr_range[0], snr_range[1], size=B).astype(np.float32)
    snr_lin = 10.0 ** (snrs[:, None, None] / 10.0)
    noise_pwr = sig_pwr / (snr_lin + 1e-12)
    noise = rng.randn(B, C, L).astype(np.float32) * np.sqrt(noise_pwr).astype(np.float32)
    return (batch + noise).astype(np.float32)


# =============================================================================
# IMS file loading + windowing
# =============================================================================
def load_ims_files(set_path: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(set_path, "*")))
    valid = [f for f in files if os.path.isfile(f) and os.path.basename(f)[:1].isdigit()]
    return valid


def read_and_window(file_path: str,
                    ch_indices: List[int],
                    window_size: int,
                    stride: int,
                    is_set1: bool) -> List[np.ndarray]:
    try:
        df = pd.read_csv(file_path, sep=r"\s+|\t+", header=None, engine="python")
        data = df.values.astype(np.float32)

        if is_set1:
            sig_x = data[:, ch_indices[0]]
            sig_y = data[:, ch_indices[1]]
            sig_z = 0.5 * (sig_x + sig_y)
            combined = np.stack([sig_x, sig_y, sig_z], axis=0)
        else:
            sig = data[:, ch_indices[0]]
            combined = np.stack([sig, sig, sig], axis=0)

        n_points = combined.shape[1]
        wins: List[np.ndarray] = []
        for start in range(0, n_points - window_size + 1, stride):
            seg = combined[:, start:start + window_size]

            # mu = np.mean(seg, axis=1, keepdims=True)
            # std = np.std(seg, axis=1, keepdims=True) + 1e-6
            # seg = (seg - mu) / std

            wins.append(seg.astype(np.float32))

        return wins
    except Exception as e:
        logger.warning(f"Corrupt/Unreadable file: {file_path} | {e}")
        return []


# =============================================================================
# Leakage-safe file-level split
# =============================================================================
def split_files(files: List[str],
                ratios: Tuple[float, float, float],
                seed: int) -> Tuple[List[str], List[str], List[str]]:
    if len(files) == 0:
        return [], [], []

    rng = np.random.RandomState(seed)
    files = list(files)
    rng.shuffle(files)

    n = len(files)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))

    n_train = max(1, min(n_train, n - 2)) if n >= 3 else max(1, n_train)
    n_val = max(1, min(n_val, n - n_train - 1)) if n - n_train >= 2 else max(0, n_val)
    n_test = n - n_train - n_val
    if n_test == 0 and n >= 2:
        n_test = 1
        n_train = max(1, n_train - 1)

    train_f = files[:n_train]
    val_f = files[n_train:n_train + n_val]
    test_f = files[n_train + n_val:]
    return train_f, val_f, test_f


def build_split_from_files(file_lists_by_class: Dict[int, List[str]],
                           ratios: Tuple[float, float, float],
                           seed: int) -> Dict[str, Dict[int, List[str]]]:
    split = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}
    for lbl, files in file_lists_by_class.items():
        tr, va, te = split_files(files, ratios, seed + 31 * lbl)
        split["train"][lbl] = tr
        split["val"][lbl] = va
        split["test"][lbl] = te
    return split


# =============================================================================
# Main preprocessing (FINAL FIX: supports ims_dirs)
# =============================================================================
def prepro_ims_full(ims_root: str,
                    ims_dirs: Optional[List[str]],
                    out_dir: str,
                    length: int = 2048,
                    stride: int = 2048,
                    seed: int = 42,
                    split_ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
                    cap_per_split_per_class: Optional[int] = None,
                    snr_benchmarks: Optional[List[int]] = None,
                    filter_epochs: int = 3) -> str:

    set_global_seed(seed, deterministic_cudnn=True)
    rng = np.random.RandomState(seed)

    logger.info(f"Device: {DEVICE}")

    # ------------------- Resolve dataset set paths -------------------
    def norm(p: str) -> str:
        return str(Path(p).expanduser().resolve())

    set_paths: Dict[str, str] = {}

    if ims_dirs and len(ims_dirs) > 0:
        ims_dirs_n = [norm(d) for d in ims_dirs]
        # map by folder name
        for d in ims_dirs_n:
            name = Path(d).name.lower()
            if "1st_test" in name:
                set_paths["Set1"] = d
            elif "2nd_test" in name:
                set_paths["Set2"] = d
            elif "4th_test" in name or "4th" in name:
                set_paths["Set3"] = d

        logger.info("Using explicit --ims_dirs:")
        for k in ["Set1", "Set2", "Set3"]:
            logger.info(f"  {k}: {set_paths.get(k, '[MISSING]')}")
    else:
        ims_root_n = norm(ims_root)
        logger.info(f"Using --ims_root: {ims_root_n}")
        set_paths["Set1"] = os.path.join(ims_root_n, IMS_CONFIG["Set1"]["folder"])
        set_paths["Set2"] = os.path.join(ims_root_n, IMS_CONFIG["Set2"]["folder"])
        set_paths["Set3"] = os.path.join(ims_root_n, IMS_CONFIG["Set3"]["folder"])

    # ------------------- Build file lists per class -------------------
    logger.info(">>> Building file lists per class (no window extraction yet)...")
    file_lists_by_class: Dict[int, List[str]] = {0: [], 1: [], 2: [], 3: []}

    # Set1
    path_s1 = set_paths.get("Set1", "")
    if path_s1 and os.path.exists(path_s1):
        files_s1 = load_ims_files(path_s1)
        logger.info(f"Set1 files: {len(files_s1)}")
        if len(files_s1) > 0:
            h_cut = int(IMS_CONFIG["Set1"]["healthy_files_ratio"] * len(files_s1))
            f_cut = int((1.0 - IMS_CONFIG["Set1"]["fault_files_ratio"]) * len(files_s1))

            healthy_files = files_s1[:max(1, h_cut)]
            fault_files = files_s1[max(0, f_cut):]

            file_lists_by_class[0].extend(healthy_files)  # Healthy from early life
            file_lists_by_class[2].extend(fault_files)    # Inner fault (B3)
            file_lists_by_class[3].extend(fault_files)    # Roller fault (B4)
    else:
        logger.warning("Set1 folder not found - cannot build Inner/Roller classes from Set1.")

    # Set2
    path_s2 = set_paths.get("Set2", "")
    if path_s2 and os.path.exists(path_s2):
        files_s2 = load_ims_files(path_s2)
        logger.info(f"Set2 files: {len(files_s2)}")
        if len(files_s2) > 0:
            h_cut = int(IMS_CONFIG["Set2"]["healthy_files_ratio"] * len(files_s2))
            f_cut = int(IMS_CONFIG["Set2"]["fault_start_ratio"] * len(files_s2))
            healthy_files = files_s2[:max(1, h_cut)]
            outer_fault_files = files_s2[max(0, f_cut):]

            file_lists_by_class[0].extend(healthy_files)
            file_lists_by_class[1].extend(outer_fault_files)
    else:
        logger.warning("Set2 folder not found (optional but recommended).")

    # Set3 (4th_test)
    path_s3 = set_paths.get("Set3", "")
    if path_s3 and os.path.exists(path_s3):
        files_s3 = load_ims_files(path_s3)
        logger.info(f"Set3 files: {len(files_s3)}")
        if len(files_s3) > 0:
            k = min(int(IMS_CONFIG["Set3"]["outer_fault_last_files"]), len(files_s3))
            outer_fault_files = files_s3[-k:]
            file_lists_by_class[1].extend(outer_fault_files)
    else:
        logger.warning("Set3 folder not found (optional).")

    # Deduplicate
    for lbl in file_lists_by_class:
        seen = set()
        uniq = []
        for f in file_lists_by_class[lbl]:
            if f not in seen:
                uniq.append(f)
                seen.add(f)
        file_lists_by_class[lbl] = uniq

    logger.info("File counts per class:")
    for lbl in sorted(file_lists_by_class):
        logger.info(f"  Class {lbl} ({CLASS_NAMES[lbl]}): {len(file_lists_by_class[lbl])} files")

    # ------------------- File-level split -------------------
    split = build_split_from_files(file_lists_by_class, split_ratios, seed)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ims_root": str(Path(ims_root).resolve()) if ims_root else "",
        "ims_dirs": ims_dirs if ims_dirs else [],
        "length": length,
        "stride": stride,
        "seed": seed,
        "split_ratios": split_ratios,
        "class_names": CLASS_NAMES,
        "split_files": {s: {str(lbl): split[s][lbl] for lbl in split[s]} for s in split}
    }

    # ------------------- Window extraction -------------------
    logger.info(">>> Extracting windows per split (file-level leakage safe)...")

    def is_set1_path(p: str) -> bool:
        p2 = p.replace("\\", "/").lower()
        return "1st_test" in p2

    def is_set3_path(p: str) -> bool:
        p2 = p.replace("\\", "/").lower()
        return "4th_test" in p2

    def extract_split(split_name: str) -> Tuple[np.ndarray, np.ndarray]:
        X_list: List[np.ndarray] = []
        y_list: List[int] = []

        for lbl in sorted(CLASS_NAMES):
            files = split[split_name][lbl]
            if len(files) == 0:
                continue

            for f in tqdm(files, desc=f"{split_name} | class {lbl}", leave=False):
                set1 = is_set1_path(f)
                set3 = is_set3_path(f)

                if set1:
                    if lbl == 0:
                        ch = [0, 1]
                    elif lbl == 2:
                        ch = [4, 5]
                    elif lbl == 3:
                        ch = [6, 7]
                    else:
                        continue
                else:
                    if set3 and lbl == 1:
                        ch = [2]
                    else:
                        ch = [0]

                wins = read_and_window(f, ch, length, stride, is_set1=set1)
                if len(wins) == 0:
                    continue
                X_list.extend(wins)
                y_list.extend([lbl] * len(wins))

        if len(X_list) == 0:
            return np.zeros((0, 3, length), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        X = np.stack(X_list, axis=0).astype(np.float32)
        y = np.array(y_list, dtype=np.int64)

        if cap_per_split_per_class is not None and cap_per_split_per_class > 0:
            X2, y2 = [], []
            for lbl2 in sorted(CLASS_NAMES):
                idx = np.where(y == lbl2)[0]
                if len(idx) == 0:
                    continue
                keep = rng.choice(idx, cap_per_split_per_class, replace=False) if len(idx) > cap_per_split_per_class else idx
                X2.append(X[keep])
                y2.append(y[keep])
            if len(X2):
                X = np.concatenate(X2, axis=0)
                y = np.concatenate(y2, axis=0)
                perm = rng.permutation(len(y))
                X, y = X[perm], y[perm]

        return X, y

    X_train, y_train = extract_split("train")
    X_val, y_val = extract_split("val")
    X_test, y_test = extract_split("test")

    logger.info(f"Samples collected | Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape}")
    logger.info(f"Train dist: {Counter(y_train)}")
    logger.info(f"Val dist:   {Counter(y_val)}")
    logger.info(f"Test dist:  {Counter(y_test)}")

    for lbl in sorted(CLASS_NAMES):
        if np.sum(y_train == lbl) == 0:
            logger.warning(f"[WARN] Train split has 0 samples for class {lbl} ({CLASS_NAMES[lbl]}).")

    # ------------------- Learned filter training (train only) -------------------
    logger.info(">>> Training learned denoising filters (train-only)...")
    filters: List[LearnedFilter] = []
    for c in range(3):
        logger.info(f"Training filter for channel {c}...")
        lf = train_filter(LearnedFilter(length), X_train[:, c, :], epochs=filter_epochs, seed=seed + 7 * c)
        filters.append(lf)

        with torch.no_grad():
            for _, ds in [("train", X_train), ("val", X_val), ("test", X_test)]:
                for i in range(0, len(ds), 256):
                    batch = torch.from_numpy(ds[i:i + 256, c, :]).float().to(DEVICE)
                    ds[i:i + 256, c, :] = lf(batch).cpu().numpy().astype(np.float32)

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ------------------- Feature extraction -------------------
    logger.info(">>> Feature extraction (GPU)...")
    extractor = GPUFeatureExtractor(length, DEVICE).eval()

    def process_batch(X: np.ndarray, augment: bool = False, snr: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        out_x, out_f = [], []
        bs = 256
        local_rng = rng

        for i in range(0, len(X), bs):
            batch = X[i:i + bs].copy()

            if augment:
                batch = apply_augmentations_and_wgn(batch, local_rng)
            elif snr is not None:
                sig_pwr = np.mean(batch ** 2, axis=2, keepdims=True) + 1e-12
                snr_lin = 10.0 ** (snr / 10.0)
                noise = local_rng.randn(*batch.shape).astype(np.float32) * np.sqrt(sig_pwr / (snr_lin + 1e-12)).astype(np.float32)
                batch = (batch + noise).astype(np.float32)

            feats = []
            for c in range(3):
                x_c = torch.from_numpy(batch[:, c, :]).float()
                f_c = extractor.extract(x_c).cpu().numpy().astype(np.float32)
                feats.append(f_c)

            out_x.append(batch.astype(np.float32))
            out_f.append(np.concatenate(feats, axis=1).astype(np.float32))

        return np.concatenate(out_x, axis=0), np.concatenate(out_f, axis=0)

    logger.info(">>> Generating final tensors (augment train only)...")
    Xt, Et = process_batch(X_train, augment=True)
    Xv, Ev = process_batch(X_val, augment=False, snr=0.0)
    Xte, Ete = process_batch(X_test, augment=False, snr=None)

    # ------------------- Feature normalization (fit on TRAIN only) -------------------
    scaler = StandardScaler().fit(Et)
    Et = scaler.transform(Et).astype(np.float32)
    Ev = scaler.transform(Ev).astype(np.float32)
    Ete = scaler.transform(Ete).astype(np.float32)

    # ------------------- Save -------------------
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "processed_ims_full.npz")

    save_dict: Dict[str, Any] = {
        "X_train_sig": Xt.astype(np.float32),
        "X_val_sig": Xv.astype(np.float32),
        "X_test_sig": Xte.astype(np.float32),
        "X_train_feat": Et,
        "X_val_feat": Ev,
        "X_test_feat": Ete,
        "y_train": y_train.astype(np.int64),
        "y_val": y_val.astype(np.int64),
        "y_test": y_test.astype(np.int64),
        "classes": CLASS_NAMES,
        "manifest_json": json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
    }

    if snr_benchmarks is None:
        snr_benchmarks = [-10, -8, -6, -4, -2]

    for snr in snr_benchmarks:
        xn, en = process_batch(X_test, augment=False, snr=float(snr))
        save_dict[f"test_snr_{snr}_sig"] = xn.astype(np.float32)
        save_dict[f"test_snr_{snr}_feat"] = scaler.transform(en).astype(np.float32)

    np.savez_compressed(save_path, **save_dict)

    for c, lf in enumerate(filters):
        torch.save(lf.state_dict(), os.path.join(out_dir, f"learned_filter_ims_ch{c}.pt"))

    logger.info(f"DONE Saved: {save_path}")
    logger.info(f"Final shapes | Train: {Xt.shape} | Val: {Xv.shape} | Test: {Xte.shape}")
    return save_path


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leakage-safe preprocessing for IMS dataset.")

    p.add_argument(
        "--ims_root",
        type=str,
        default=r"D:\Bearing Project\4. Bearings\IMS",
        help="Root folder containing 1st_test/2nd_test/4th_test (default: your IMS root).",
    )

    p.add_argument(
        "--ims_dirs",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional explicit paths to IMS set folders "
            r"(e.g., D:\...\IMS\1st_test D:\...\IMS\2nd_test D:\...\IMS\4th_test). "
            "If provided, overrides --ims_root."
        ),
    )

    p.add_argument("--out_dir", type=str, default="", help="Output directory for .npz and filter weights")
    p.add_argument("--length", type=int, default=2048, help="Window length")
    p.add_argument("--stride", type=int, default=2048, help="Window stride (use =length for non-overlap)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--cap", type=int, default=0, help="Cap per split per class (0 disables)")
    p.add_argument("--filter_epochs", type=int, default=3, help="Epochs for learned filter training (train-only)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    out_dir = (args.out_dir or "").strip()
    if not out_dir:
        # default output near IMS root (or cwd)
        base = Path(args.ims_root).expanduser().resolve() if args.ims_root else Path.cwd()
        out_dir = str(base / "processed_ims")

    cap = None if int(args.cap) <= 0 else int(args.cap)

    prepro_ims_full(
        ims_root=str(args.ims_root).strip(),
        ims_dirs=args.ims_dirs,
        out_dir=out_dir,
        length=int(args.length),
        stride=int(args.stride),
        seed=int(args.seed),
        cap_per_split_per_class=cap,
        filter_epochs=int(args.filter_epochs),
    )

# =================================
p = r"D:\Bearing Project\4. Bearings\IMS\processed_ims\processed_ims_full.npz"
data = np.load(p, allow_pickle=True)
print("Keys:", data.files)
for k in data.files:
    v = data[k]
    if hasattr(v, "shape"):
        print(k, v.shape, v.dtype)
    else:
        print(k, type(v))
