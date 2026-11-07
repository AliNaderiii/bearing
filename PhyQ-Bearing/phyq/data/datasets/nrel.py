"""NREL Wind Turbine Gearbox Condition Monitoring (healthy H* vs damaged D*).

Two layouts are supported:
1. CSV files whose columns are named ``{exp}_AN{n}`` (legacy layout, e.g. ``D1_AN3``);
2. one file per experiment (``H1.mat`` / ``D3.mat`` / ``H1.csv``) with channels
   ``AN3..AN10`` as variables/columns.
Experiment = physical run -> ``group``; splits are experiment-wise.
"""
from __future__ import annotations

import glob
import os
import re
from functools import partial
from typing import Dict, List

import numpy as np

from ..recording import Recording

FS = 40000.0


def _stage_csv(path: str, staging: str) -> str:
    """Read a (possibly multi-GB) CSV ONCE, in chunks, and store each *_AN<n> column as .npy in the
    project cache. The original CSV is opened read-only and never modified."""
    import hashlib
    import pandas as pd
    tag = hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:8]
    out = os.path.join(staging, f"{os.path.splitext(os.path.basename(path))[0]}_{tag}")
    if os.path.exists(os.path.join(out, "_done")):
        return out
    os.makedirs(out, exist_ok=True)
    cols = [c for c in pd.read_csv(path, nrows=0).columns if re.search(r"AN\d+$", str(c))]
    parts: Dict[str, list] = {c: [] for c in cols}
    for chunk in pd.read_csv(path, usecols=cols, chunksize=500_000):
        for c in cols:
            parts[c].append(chunk[c].to_numpy(dtype=np.float32))
    for c in cols:
        np.save(os.path.join(out, f"{c}.npy"), np.concatenate(parts[c]))
        parts[c] = None
    open(os.path.join(out, "_done"), "w").close()
    return out


def _load_csv_cols(path: str, cols: List[str], staging: str) -> np.ndarray:
    d = _stage_csv(path, staging)
    out = []
    for c in cols:
        o = np.load(os.path.join(d, f"{c}.npy"))
        fin = np.isfinite(o)
        out.append(o[: (np.where(fin)[0].max() + 1) if fin.any() else 0])  # drop trailing NaN padding
    n = min(len(o) for o in out)
    return np.stack([o[:n] for o in out], 0)


def _load_mat(path: str, chans: List[str]) -> np.ndarray:
    import scipy.io as sio
    m = sio.loadmat(path)
    out = [np.asarray(m[c], dtype=np.float32).ravel() for c in chans]
    n = min(len(o) for o in out)
    return np.stack([o[:n] for o in out], 0)


def _exp_key(e: str):
    return (e[0], int(re.sub(r"\D", "", e) or 0))


def build(cfg: dict) -> List[Recording]:
    import pandas as pd
    root = cfg["root"]
    want = cfg.get("channels")  # e.g. ["AN3", ..., "AN10"]; None -> all found
    found: Dict[str, tuple] = {}
    for p in sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)):
        try:
            cols = list(pd.read_csv(p, nrows=0).columns)
        except Exception:
            continue
        exps = sorted({c.split("_AN")[0] for c in cols if "_AN" in c})
        if exps:
            for e in exps:
                ec = sorted([c for c in cols if c.startswith(f"{e}_AN")], key=lambda c: int(c.split("_AN")[-1]))
                if want:
                    ec = [f"{e}_{w}" for w in want if f"{e}_{w}" in ec]
                found[e] = ("csv", p, ec)
        else:
            stem = os.path.splitext(os.path.basename(p))[0]
            if re.fullmatch(r"[HD]\d+", stem):
                ec = sorted([c for c in cols if re.fullmatch(r"AN\d+", c)], key=lambda c: int(c[2:]))
                if want:
                    ec = [w for w in want if w in ec]
                found[stem] = ("csv", p, ec)
    for p in sorted(glob.glob(os.path.join(root, "**", "*.mat"), recursive=True)):
        stem = os.path.splitext(os.path.basename(p))[0]
        if re.fullmatch(r"[HD]\d+", stem) and stem not in found:
            import scipy.io as sio
            names = sorted([w[0] for w in sio.whosmat(p) if re.fullmatch(r"AN\d+", w[0])], key=lambda c: int(c[2:]))
            if want:
                names = [w for w in want if w in names]
            found[stem] = ("mat", p, names)
    recs = []
    for e in sorted(found, key=_exp_key):
        kind, p, cols = found[e]
        if not cols:
            continue
        lbl, name = (0, "Healthy") if e.startswith("H") else (1, "Damaged")
        staging = os.path.join(cfg.get("cache_dir", "cache"), "nrel", "_staging")
        loader = partial(_load_csv_cols, p, cols, staging) if kind == "csv" else partial(_load_mat, p, cols)
        recs.append(Recording(rid=f"nrel_{e}", dataset="nrel", label=lbl, label_name=name, group=e,
                              condition="all", fs=float(cfg.get("fs", FS)), loader=loader, channels=cols,
                              meta={"path": p, "experiment": e}))
    n_ch = {len(r.channels) for r in recs}
    if len(n_ch) > 1:
        raise RuntimeError(f"NREL experiments have different channel counts {n_ch}; set data.channels")
    return recs
