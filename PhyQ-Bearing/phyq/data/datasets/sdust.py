"""SDUST bearing dataset (generic recursive loader).

Labels are inferred from path tokens exactly as in the legacy
``Preprocess_SDUST.py`` (nc -> Normal; if0.2/0.4/0.6 -> IF; of* -> OF; rf* -> RF),
but the split is chronological per file (or cross-condition) instead of the
legacy random window shuffle, which leaked overlapping windows across splits.

The operating condition is parsed with ``condition_regex`` (configurable;
default matches e.g. ``1000rpm`` / ``2000rpm_10Nm``) applied to the relative path.
"""
from __future__ import annotations

import glob
import os
import re
from functools import partial
from typing import List

import numpy as np

from ..recording import Recording

CLASSES = ["NC", "IF0.2", "IF0.4", "IF0.6", "OF0.2", "OF0.4", "OF0.6", "RF0.2", "RF0.4", "RF0.6"]


def infer_label(path: str):
    top = path.replace("\\", "/").split("/")[0].upper()
    if top in CLASSES:                      # e.g. <root>/IF0.2/<file>
        return CLASSES.index(top)
    s = path.lower().replace("\\", "/")
    toks = re.split(r"[/_\-\s]+", s)
    joined = " ".join(toks)
    if re.search(r"(^|[ /])nc($|[ ./])", joined) or "normal" in s:
        return 0
    for base, kind in ((1, "if"), (4, "of"), (7, "rf")):
        m = re.search(kind + r"\s*0?\.?([246])", joined)
        if m:
            return base + {"2": 0, "4": 1, "6": 2}[m.group(1)]
    return None


def _numeric_arrays(obj, out, depth=0):
    if depth > 8:
        return
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in "fiu" and obj.size >= 4096:
            a = np.asarray(obj, dtype=np.float32)
            if a.ndim == 1:
                out.append(a)
            elif a.ndim == 2:
                for row in (a.T if a.shape[0] > a.shape[1] else a):
                    out.append(np.asarray(row).ravel())
        elif obj.dtype == object:
            for o in obj.ravel():
                _numeric_arrays(o, out, depth + 1)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if not str(k).startswith("__"):
                _numeric_arrays(v, out, depth + 1)
    elif hasattr(obj, "_fieldnames"):
        for k in obj._fieldnames:
            _numeric_arrays(getattr(obj, k), out, depth + 1)


def _clean_channels(arrs):
    """Drop constant arrays, monotonic time/index vectors and exact duplicates.
    (The legacy code silently DUPLICATED a channel when fewer than 3 existed; we never do.)"""
    out = []
    for a in arrs:
        a = np.asarray(a, dtype=np.float32).ravel()
        if a.size < 4096 or not np.isfinite(a).all() or np.std(a) == 0:
            continue
        d = np.diff(a[:5000])
        if (d > 0).all() or (d < 0).all():
            continue
        if any(len(b) == len(a) and np.array_equal(b, a) for b in out):
            continue
        out.append(a)
    return out


def _load(path: str, n_ch) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mat":
        import scipy.io as sio
        arrs: list = []
        _numeric_arrays(sio.loadmat(path, squeeze_me=True, struct_as_record=False), arrs)
    else:
        import pandas as pd
        df = pd.read_csv(path, sep=None, engine="python", header=None)
        df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").dropna(axis=0)
        a = df.to_numpy(np.float32)
        arrs = [a[:, j] for j in range(a.shape[1])]
    arrs = _clean_channels(arrs)
    if not arrs:
        raise ValueError(f"{path}: no signal found")
    if n_ch not in (None, "auto"):
        if len(arrs) < int(n_ch):
            raise ValueError(f"{path}: found {len(arrs)} real channels, need {n_ch} (never duplicated)")
        arrs = arrs[: int(n_ch)]
    n = min(len(a) for a in arrs)
    return np.stack([a[:n] for a in arrs], 0)


def build(cfg: dict) -> List[Recording]:
    root = cfg["root"]
    n_ch = cfg.get("n_channels", "auto")
    rx = re.compile(cfg.get("condition_regex", r"(\d+\s*rpm(?:[_\- ]*\d+\s*n\.?m)?)"), re.I)
    files = []
    for ext in ("mat", "csv", "txt"):
        files += glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True)
    recs = []
    for p in sorted(files):
        rel = os.path.relpath(p, root)
        lbl = infer_label(rel)
        if lbl is None:
            continue
        m = rx.search(rel)
        cond = m.group(1).lower().replace(" ", "") if m else "all"
        recs.append(Recording(
            rid=f"sdust_{rel}", dataset="sdust", label=lbl, label_name=CLASSES[lbl], group=CLASSES[lbl],
            condition=cond, fs=float(cfg.get("fs", 25600.0)), loader=partial(_load, p, n_ch),
            channels=[], meta={"path": p},
        ))
    return recs
