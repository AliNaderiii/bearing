# -*- coding: utf-8 -*-
r"""
Preprocess_NREL.py
Author: Ali Naderi - 2026 (Leak-Free + Paper-Ready Edition) [FIXED DIMS]

Outputs (NPZ):
  X_train_sig, X_val_sig, X_test_sig : (N, 8, 2048) float32
  X_train_feat, X_val_feat, X_test_feat : (N, 88) float32   (11 feats * 8 channels)
  y_train, y_val, y_test : (N,) int64
  classes : object array ["Healthy","Damaged"]
  manifest_json : bytes
  test_snr_{snr}_sig / test_snr_{snr}_feat for robustness
"""

from __future__ import annotations

import os, sys, json, glob, time, math, random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 0) CONFIG (VSCode Friendly)
# =============================================================================
@dataclass
class CFG:
    root_dir: str = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\Wind Turbine Gearbox Condition Monitoring Vibration\Data Information"
    out_npz: str = "nrel_processed_robust.npz"

    target_sr: int = 40000
    window: int = 2048
    stride_train: int = 1024
    stride_val: int = 2048
    stride_test: int = 2048

    train_exps: List[str] = None
    val_exps:   List[str] = None
    test_exps:  List[str] = None

    instance_norm: bool = True
    clamp_after_norm: float = 0.0  # 0 disables
    skip_nan_inf: bool = True

    test_snrs: List[float] = None
    noise_seed: int = 123

    feat_batch: int = 256
    device: str = "cuda"

    preprocess_seed: int = 2026


def build_default_cfg() -> CFG:
    cfg = CFG()
    cfg.train_exps = ["D3","D4","D5","D7","D8","H1","H2","H3","H4","H5","H6","H8"]
    cfg.val_exps   = ["D1","D2","D6","H7"]
    cfg.test_exps  = ["D9","D10","H9","H10"]
    cfg.test_snrs  = [-10.0, -8.0, -6.0, -4.0, -2.0]
    return cfg


# =============================================================================
# 1) LOG + SEED
# =============================================================================
def log(msg: str) -> None:
    print(msg, flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# 2) NOISE (WGN)
# =============================================================================
def wgn_1d(x: np.ndarray, snr_db: float, rng: np.random.RandomState) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    snr_lin = 10.0 ** (snr_db / 10.0)
    xpower = float(np.mean(x * x))
    npower = xpower / max(snr_lin, 1e-12)
    noise = rng.randn(x.size) * math.sqrt(npower)
    return (x + noise).astype(np.float32)


def add_noise_dataset(X: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed + int(round(snr_db * 10)))
    N, C, L = X.shape
    out = np.empty_like(X, dtype=np.float32)
    for i in range(N):
        for c in range(C):
            out[i, c] = wgn_1d(X[i, c], snr_db, rng)
    return out


def snr_key(snr: float) -> str:
    if abs(snr - int(snr)) < 1e-9:
        return str(int(snr))
    s = f"{snr:.2f}".rstrip("0").rstrip(".")
    return s


# =============================================================================
# 3) FEATURE EXTRACTOR (GPU)  -- FIXED DIMS
# =============================================================================
class GPUFeatureExtractor(nn.Module):
    """
    Input: x (B, L)
    Output: (B, 11)  [ALL features strictly 2D]
    """
    def __init__(self, fs: int, length: int, nperseg: int = 256):
        super().__init__()
        self.fs = fs
        self.length = length
        self.nperseg = int(nperseg)
        self.hop = max(1, self.nperseg // 4)
        self.register_buffer("hann", torch.hann_window(self.nperseg), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,L)
        B, L = x.shape
        window = self.hann.to(x.device)

        # Time domain
        mean_t = torch.mean(x, dim=1, keepdim=True)                       # (B,1)
        std_t  = torch.std(x, dim=1, keepdim=True) + 1e-9                # (B,1)
        rms    = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-12)  # (B,1)
        peak   = torch.max(torch.abs(x), dim=1, keepdim=True)[0]         # (B,1)
        crest  = peak / (rms + 1e-9)                                     # (B,1)

        z = (x - mean_t) / std_t
        skew = torch.mean(z ** 3, dim=1, keepdim=True)                   # (B,1)
        kurt = torch.mean(z ** 4, dim=1, keepdim=True)                   # (B,1)

        shape_factor = rms / (torch.mean(torch.abs(x), dim=1, keepdim=True) + 1e-9)  # (B,1)

        # Frequency domain (frame-wise energy)
        x_pad  = F.pad(x, (0, self.nperseg))
        frames = x_pad.unfold(1, self.nperseg, self.hop)                 # (B, n_frames, nperseg)
        win    = frames * window.view(1, 1, -1)
        fft    = torch.fft.rfft(win, dim=2)
        mag    = torch.abs(fft)
        spec_energy = torch.mean(torch.sum(mag ** 2, dim=2), dim=1, keepdim=True) / (self.length + 1e-12)  # (B,1)

        # Envelope (Hilbert-like)
        fft_full = torch.fft.fft(win, dim=2)
        h = torch.zeros(self.nperseg, device=x.device, dtype=fft_full.dtype)
        h[0] = 1
        h[1:self.nperseg // 2] = 2
        env = torch.abs(torch.fft.ifft(fft_full * h, dim=2))             # (B, n_frames, nperseg)

        # IMPORTANT: keep outputs (B,1)
        env_mean = env.mean(dim=(1, 2), keepdim=False).unsqueeze(1)      # (B,1)
        env_std  = env.std(dim=2).mean(dim=1, keepdim=True)             # (B,1)

        feats = torch.cat([
            mean_t, std_t, rms, skew, kurt, peak,
            crest, shape_factor, spec_energy, env_mean, env_std
        ], dim=1)                                                       # (B,11)

        return feats


@torch.no_grad()
def extract_features_88(extractor: GPUFeatureExtractor, X: np.ndarray, device: torch.device, bs: int) -> np.ndarray:
    """
    X: (N, C, L) float32
    returns: (N, 11*C) float32  -> for C=8 => 88
    """
    N, C, L = X.shape
    extractor.eval()
    out_list = []
    for i in tqdm(range(0, N, bs), desc="FeatExtract"):
        batch = X[i:i + bs]
        B, C, L = batch.shape
        flat = torch.from_numpy(batch.reshape(B * C, L)).to(device=device, dtype=torch.float32)
        feats11 = extractor(flat)  # (B*C, 11)
        feats = feats11.reshape(B, C * 11).contiguous().cpu().numpy().astype(np.float32)
        out_list.append(feats)
    return np.concatenate(out_list, axis=0)


# =============================================================================
# 4) WINDOWING + SPLIT
# =============================================================================
def list_csv_files(root_dir: str) -> List[str]:
    files = glob.glob(os.path.join(root_dir, "*.csv"))
    return sorted(files)


def find_experiments_from_columns(cols: List[str]) -> List[str]:
    exps = set()
    for c in cols:
        if "_AN" in c:
            exps.add(c.split("_AN")[0])
    return sorted(exps)


def exp_label(exp: str) -> int:
    return 0 if exp.startswith("H") else 1


def pick_split(exp: str, cfg: CFG) -> Optional[Tuple[str, int]]:
    if exp in cfg.train_exps:
        return ("train", cfg.stride_train)
    if exp in cfg.val_exps:
        return ("val", cfg.stride_val)
    if exp in cfg.test_exps:
        return ("test", cfg.stride_test)
    return None


def stable_instance_norm(seg: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = np.mean(seg, axis=1, keepdims=True)
    std = np.std(seg, axis=1, keepdims=True) + eps
    return (seg - mean) / std


def iter_windows_from_matrix(data_matrix: np.ndarray, win: int, stride: int) -> List[np.ndarray]:
    C, T = data_matrix.shape
    if T < win:
        return []
    n = 1 + (T - win) // stride
    out = []
    for i in range(n):
        s = i * stride
        e = s + win
        out.append(data_matrix[:, s:e])
    return out


# =============================================================================
# 5) MAIN PROCESSOR
# =============================================================================
def process_nrel(cfg: CFG) -> Dict[str, Any]:
    seed_everything(cfg.preprocess_seed)
    device = torch.device("cuda" if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")

    csv_files = list_csv_files(cfg.root_dir)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {cfg.root_dir}")

    log("=" * 80)
    log(f"[NREL] CSV files: {len(csv_files)}")
    log(f"[SPLIT] TRAIN exps ({len(cfg.train_exps)}): {sorted(cfg.train_exps)}")
    log(f"[SPLIT] VAL   exps ({len(cfg.val_exps)}): {sorted(cfg.val_exps)}")
    log(f"[SPLIT] TEST  exps ({len(cfg.test_exps)}): {sorted(cfg.test_exps)}")
    log(f"[PARAM] WIN={cfg.window} stride_train={cfg.stride_train} stride_val={cfg.stride_val} stride_test={cfg.stride_test}")
    log(f"[NORM ] instance_norm={cfg.instance_norm} clamp={cfg.clamp_after_norm} skip_nan_inf={cfg.skip_nan_inf}")
    log(f"[NOISE] test_snrs={cfg.test_snrs}")
    log("=" * 80)

    store = {
        "train": {"sig": [], "lbl": [], "exp": []},
        "val":   {"sig": [], "lbl": [], "exp": []},
        "test":  {"sig": [], "lbl": [], "exp": []},
    }

    per_exp_counts: Dict[str, Dict[str, int]] = {}

    for fp in csv_files:
        fname = os.path.basename(fp)
        log(f"\n[SCAN] {fname}")
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            log(f"[WARN] cannot read {fname}: {e}")
            continue

        cols = list(df.columns)
        exps = find_experiments_from_columns(cols)
        if not exps:
            log("[WARN] no experiments found in columns (no *_AN*).")
            continue

        for exp in tqdm(exps, desc="Experiments"):
            sp = pick_split(exp, cfg)
            if sp is None:
                continue
            split, stride = sp

            exp_cols = [c for c in cols if c.startswith(f"{exp}_AN")]
            if not exp_cols:
                continue

            try:
                exp_cols.sort(key=lambda x: int(x.split("_AN")[-1]))
            except Exception:
                exp_cols.sort()

            mat = df[exp_cols].values.T  # (C,T)
            if mat.ndim != 2:
                continue

            windows = iter_windows_from_matrix(mat, cfg.window, stride)
            if not windows:
                continue

            y = exp_label(exp)
            kept = 0

            for seg in windows:
                seg = seg.astype(np.float32, copy=False)

                if cfg.instance_norm:
                    seg = stable_instance_norm(seg)

                if cfg.clamp_after_norm and cfg.clamp_after_norm > 0:
                    seg = np.clip(seg, -cfg.clamp_after_norm, cfg.clamp_after_norm)

                if cfg.skip_nan_inf and (not np.isfinite(seg).all()):
                    continue

                store[split]["sig"].append(seg)
                store[split]["lbl"].append(y)
                store[split]["exp"].append(exp)
                kept += 1

            if exp not in per_exp_counts:
                per_exp_counts[exp] = {"train": 0, "val": 0, "test": 0}
            per_exp_counts[exp][split] += kept

    def pack(split: str) -> Tuple[np.ndarray, np.ndarray]:
        X = np.stack(store[split]["sig"]).astype(np.float32)
        y = np.asarray(store[split]["lbl"], dtype=np.int64)
        return X, y

    Xtr, ytr = pack("train")
    Xva, yva = pack("val")
    Xte, yte = pack("test")

    def counts(y: np.ndarray) -> List[int]:
        uniq = sorted(set(y.tolist()))
        return [int((y == k).sum()) for k in uniq]

    log("\n" + "-" * 80)
    log(f"[SHAPES] Xtr={Xtr.shape} ytr={ytr.shape} | Xva={Xva.shape} yva={yva.shape} | Xte={Xte.shape} yte={yte.shape}")
    log(f"[DISTR] train counts={counts(ytr)}")
    log(f"[DISTR] val   counts={counts(yva)}")
    log(f"[DISTR] test  counts={counts(yte)}")
    log("-" * 80)

    extractor = GPUFeatureExtractor(fs=cfg.target_sr, length=cfg.window).to(device)

    log("[FEAT] extracting TRAIN feats...")
    Ftr = extract_features_88(extractor, Xtr, device, cfg.feat_batch)

    log("[FEAT] extracting VAL feats...")
    Fva = extract_features_88(extractor, Xva, device, cfg.feat_batch)

    log("[FEAT] extracting TEST feats (clean)...")
    Fte = extract_features_88(extractor, Xte, device, cfg.feat_batch)

    classes = ["Healthy", "Damaged"]

    manifest = {
        "dataset": "NREL Wind Turbine Gearbox CM Vibration (CSV)",
        "root_dir": cfg.root_dir,
        "csv_files": [os.path.basename(x) for x in csv_files],
        "split": {
            "train_exps": sorted(cfg.train_exps),
            "val_exps": sorted(cfg.val_exps),
            "test_exps": sorted(cfg.test_exps),
            "stride_train": cfg.stride_train,
            "stride_val": cfg.stride_val,
            "stride_test": cfg.stride_test,
            "window": cfg.window,
            "target_sr": cfg.target_sr,
        },
        "preprocess": {
            "instance_norm": cfg.instance_norm,
            "clamp_after_norm": cfg.clamp_after_norm,
            "skip_nan_inf": cfg.skip_nan_inf,
            "seed": cfg.preprocess_seed,
        },
        "shapes": {
            "X_train_sig": list(Xtr.shape),
            "X_val_sig": list(Xva.shape),
            "X_test_sig": list(Xte.shape),
            "X_train_feat": list(Ftr.shape),
            "X_val_feat": list(Fva.shape),
            "X_test_feat": list(Fte.shape),
        },
        "per_experiment_window_counts": per_exp_counts,
        "note": "Experiment-level split prevents leakage; Train overlap, Val/Test no overlap.",
    }
    manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")

    out: Dict[str, Any] = {
        "X_train_sig": Xtr,
        "X_val_sig": Xva,
        "X_test_sig": Xte,
        "X_train_feat": Ftr,
        "X_val_feat": Fva,
        "X_test_feat": Fte,
        "y_train": ytr,
        "y_val": yva,
        "y_test": yte,
        "classes": np.array(classes, dtype=object),
        "manifest_json": manifest_json,
    }

    for snr in cfg.test_snrs:
        k = snr_key(float(snr))
        log(f"[NOISE] generating test_snr_{k} ...")
        X_noisy = add_noise_dataset(Xte, float(snr), seed=cfg.noise_seed)
        F_noisy = extract_features_88(extractor, X_noisy, device, cfg.feat_batch)
        out[f"test_snr_{k}_sig"] = X_noisy
        out[f"test_snr_{k}_feat"] = F_noisy

    return out

def main():
    cfg = build_default_cfg()

    t0 = time.time()
    data = process_nrel(cfg)
    np.savez(cfg.out_npz, **data)
    dt = time.time() - t0

    log("\n" + "=" * 80)
    log(f"[SUCCESS] Saved: {cfg.out_npz}")
    log(f"Train: {len(data['y_train'])} | Val: {len(data['y_val'])} | Test: {len(data['y_test'])}")
    log(f"Signal shape train: {data['X_train_sig'].shape} | feat_dim={data['X_train_feat'].shape[1]}")
    log(f"Noisy test sets: {cfg.test_snrs}")
    log(f"Time: {dt:.1f} sec")
    log("=" * 80)

if __name__ == "__main__":
    main()