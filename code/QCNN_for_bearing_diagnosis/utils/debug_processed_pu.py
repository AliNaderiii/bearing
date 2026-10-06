# -*- coding: utf-8 -*-
"""
debug_processed_pu.py  (SAFE PRINT + EXTRA CHECKS)
=========================================================
Deep Debug & Sanity Inspector for Processed PU Dataset
- Windows-safe printing (no Unicode crash)
- Saves full report to TXT file
- Extra checks: scale-consistency, per-channel stats, feature shift
=========================================================
"""

import os
import json
import sys
import numpy as np
from collections import Counter
from datetime import datetime

# =========================
# CONFIG
# =========================
NPZ_PATH = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\Paderborn\processed_pu\processed_pu_full.npz"
META_PATH = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\Paderborn\processed_pu\meta_pu.json"
SAVE_TXT = True  # also save to a report txt next to NPZ

# =========================
# SAFE PRINT HELPERS
# =========================
class Tee:
    def __init__(self, fp=None):
        self.fp = fp
    def write(self, s):
        try:
            sys.__stdout__.write(s)
        except Exception:
            # last-resort fallback
            sys.__stdout__.write(s.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))
        if self.fp is not None:
            self.fp.write(s)
    def flush(self):
        sys.__stdout__.flush()
        if self.fp is not None:
            self.fp.flush()

def print_block(title: str):
    bar = "=" * 80
    print("\n" + bar)
    print(title)   # no unicode symbols
    print(bar)

def stat(x: np.ndarray):
    return {
        "shape": tuple(x.shape),
        "dtype": str(x.dtype),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "nan": int(np.isnan(x).sum()),
        "inf": int(np.isinf(x).sum()),
    }

def per_channel_stat(x_sig: np.ndarray):
    # x_sig: (N, 3, L)
    out = []
    for c in range(x_sig.shape[1]):
        xc = x_sig[:, c, :]
        out.append(stat(xc))
    return out

def feature_shift(train_feat: np.ndarray, feat: np.ndarray):
    # compare mean/std to train
    m_tr = np.mean(train_feat, axis=0)
    s_tr = np.std(train_feat, axis=0) + 1e-12
    m = np.mean(feat, axis=0)
    s = np.std(feat, axis=0) + 1e-12
    # normalized shifts
    mean_shift = float(np.mean(np.abs(m - m_tr)))
    std_shift  = float(np.mean(np.abs(s - s_tr)))
    # z-shift of means in train units
    z_mean = float(np.mean(np.abs((m - m_tr) / s_tr)))
    return {"mean_abs_diff": mean_shift, "std_abs_diff": std_shift, "z_mean_shift": z_mean}

# =========================
# MAIN
# =========================
def main():
    assert os.path.exists(NPZ_PATH), f"NPZ not found: {NPZ_PATH}"

    # setup txt logging
    report_fp = None
    tee = None
    if SAVE_TXT:
        base = os.path.dirname(NPZ_PATH)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_fp_path = os.path.join(base, f"PU_DEBUG_REPORT_{ts}.txt")
        report_fp = open(report_fp_path, "w", encoding="utf-8")
        tee = Tee(report_fp)
        sys.stdout = tee
        print(f"[INFO] Writing report to: {report_fp_path}")

    print_block("PU DATASET DEBUG INSPECT REPORT (SAFE PRINT)")

    data = np.load(NPZ_PATH, allow_pickle=True)
    keys = list(data.files)

    # 1) keys
    print_block("1) NPZ KEYS")
    for k in keys:
        v = data[k]
        if hasattr(v, "shape"):
            print(f"{k:25s} | shape={v.shape} dtype={v.dtype}")
        else:
            print(f"{k:25s} | <object>")

    # 2) split shapes
    print_block("2) BASIC SPLIT SHAPES")
    for split in ["train", "val", "test"]:
        xs = data[f"X_{split}_sig"]
        xf = data[f"X_{split}_feat"]
        y  = data[f"y_{split}"]
        print(f"{split.upper():5s} | sig={xs.shape} feat={xf.shape} labels={y.shape}")

    # 3) class dist
    print_block("3) CLASS DISTRIBUTIONS")
    for split in ["train", "val", "test"]:
        y = data[f"y_{split}"]
        cnt = Counter(y.tolist())
        print(f"{split.upper():5s} -> {dict(cnt)}")

    # 4) signal sanity global + per-channel
    print_block("4) SIGNAL SANITY (GLOBAL + PER-CHANNEL)")
    for split in ["train", "val", "test"]:
        xs = data[f"X_{split}_sig"]
        s = stat(xs)
        print(f"{split.upper():5s} | GLOBAL min={s['min']:.4f} max={s['max']:.4f} "
              f"mean={s['mean']:.4f} std={s['std']:.4f} NaN={s['nan']} Inf={s['inf']}")
        pcs = per_channel_stat(xs)
        for c, sc in enumerate(pcs):
            print(f"   ch{c}: min={sc['min']:.4f} max={sc['max']:.4f} mean={sc['mean']:.4f} std={sc['std']:.4f}")

    # 5) feature sanity + shift
    print_block("5) FEATURE SANITY + SHIFT VS TRAIN")
    train_feat = data["X_train_feat"]
    for split in ["train", "val", "test"]:
        xf = data[f"X_{split}_feat"]
        s = stat(xf)
        print(f"{split.upper():5s} | mean={s['mean']:.4f} std={s['std']:.4f} NaN={s['nan']} Inf={s['inf']}")
        if split != "train":
            sh = feature_shift(train_feat, xf)
            print(f"   shift: mean_abs_diff={sh['mean_abs_diff']:.4f} std_abs_diff={sh['std_abs_diff']:.4f} z_mean_shift={sh['z_mean_shift']:.4f}")

    # 6) SNR checks
    print_block("6) NOISY TEST BENCHMARKS")
    base_n = data["X_test_sig"].shape[0]
    snr_sigs = sorted([k for k in keys if k.startswith("test_snr_") and k.endswith("_sig")])
    for k in snr_sigs:
        snr = k.replace("test_snr_", "").replace("_sig", "")
        xs = data[k]
        xf = data.get(f"test_snr_{snr}_feat", None)
        ok = (xs.shape[0] == base_n) and (xf is not None) and (xf.shape[0] == base_n)
        s = stat(xs)
        print(f"SNR {snr:>4s} | OK={ok} | sig[min={s['min']:.3f}, max={s['max']:.3f}] | NaN={s['nan']} Inf={s['inf']}")

    # 7) meta
    print_block("7) META FILE CHECK")
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print("Meta keys:", list(meta.keys()))
        print("classes:", meta.get("classes"))
        print("class_names:", meta.get("class_names"))
        print("split_ratios:", meta.get("split_ratios"))
        print("counts:", meta.get("counts"))
        print("bad_files_count:", meta.get("bad_files_count"))
        print("device:", meta.get("device"))
        print("cublas_workspace_config:", meta.get("cublas_workspace_config"))
    else:
        print("meta_pu.json NOT FOUND")

    print_block("DEBUG INSPECTION FINISHED (OK)")

    # restore stdout and close report
    if report_fp is not None:
        sys.stdout = sys.__stdout__
        report_fp.close()
        print("[INFO] Report saved successfully.")

if __name__ == "__main__":
    main()
