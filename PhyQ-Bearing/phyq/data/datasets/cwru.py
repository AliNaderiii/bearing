"""CWRU 12 kHz drive-end bearing data.

Fixes vs. the legacy pipeline (see docs/LEGACY_MAPPING.md):
* variables are matched by the EXACT key ``X{id:03d}_DE_time`` (file 99.mat also
  contains X098_* keys; the legacy code silently used those);
* the base-accelerometer channel (``_BA_time``) is never used: it is absent in the
  normal-baseline files and was zero-filled by the legacy code -> trivial shortcut;
* one recording per (class, load); within-load splits are chronological.
"""
from __future__ import annotations

import glob
import os
import re
from functools import partial
from typing import Dict, List, Tuple

import numpy as np
import scipy.io as sio

from ..recording import Recording

CLASSES = ["Normal", "IR007", "B007", "OR007", "IR014", "B014", "OR014", "IR021", "B021", "OR021"]

# load (HP) -> file ids in CLASSES order (OR at 6 o'clock)
FILE_IDS: Dict[int, List[int]] = {
    0: [97, 105, 118, 130, 169, 185, 197, 209, 222, 234],
    1: [98, 106, 119, 131, 170, 186, 198, 210, 223, 235],
    2: [99, 107, 120, 132, 171, 187, 199, 211, 224, 236],
    3: [100, 108, 121, 133, 172, 188, 200, 212, 225, 237],
}
FS = 12000.0


def _index_files(root) -> Dict[int, Tuple[str, str]]:
    """Map file-id -> (path, prefix) by inspecting variable names (headers only, read-only).
    ``root`` may be one folder or a list of folders (e.g. .../0HP, .../1HP, ...)."""
    found: Dict[int, Tuple[str, str]] = {}
    roots = root if isinstance(root, (list, tuple)) else [root]
    paths = []
    for r in roots:
        paths += sorted(glob.glob(os.path.join(r, "**", "*.mat"), recursive=True))
    for p in paths:
        try:
            names = [w[0] for w in sio.whosmat(p)]
        except Exception:
            continue
        stem_digits = re.findall(r"\d+", os.path.basename(p))
        stem_id = int(stem_digits[0]) if stem_digits else None
        for n in names:
            m = re.fullmatch(r"X(\d+)_DE_time", n)
            if not m:
                continue
            fid = int(m.group(1))
            prefix = n[: -len("_DE_time")]
            # prefer the file whose name matches the id (99.mat holds X098 AND X099)
            if fid not in found or stem_id == fid:
                found[fid] = (p, prefix)
    return found


def _load(path: str, prefix: str, channels: List[str]) -> np.ndarray:
    m = sio.loadmat(path)
    out = []
    for ch in channels:
        key = f"{prefix}_{ch}_time"
        if key not in m:
            raise KeyError(f"{path}: missing exact key {key} (refusing to substitute)")
        out.append(np.asarray(m[key], dtype=np.float32).ravel())
    n = min(len(o) for o in out)
    return np.stack([o[:n] for o in out], 0)


def build(cfg: dict) -> List[Recording]:
    root = cfg["root"]
    loads = cfg.get("loads", [0, 1, 2, 3])
    channels = cfg.get("channels", ["DE"])
    if "BA" in channels:
        raise ValueError("BA channel is disabled: it is missing in normal files (shortcut).")
    idx = _index_files(root)
    recs: List[Recording] = []
    missing = []
    for ld in loads:
        for lbl, fid in enumerate(FILE_IDS[int(ld)]):
            if fid not in idx:
                missing.append(fid)
                continue
            path, prefix = idx[fid]
            recs.append(Recording(
                rid=f"cwru_{fid:03d}", dataset="cwru", label=lbl, label_name=CLASSES[lbl],
                group=CLASSES[lbl], condition=f"{ld}HP", fs=FS,
                loader=partial(_load, path, prefix, channels), channels=channels,
                meta={"file_id": fid, "path": path, "key_prefix": prefix},
            ))
    if missing and cfg.get("strict", True):
        by_load = {f"{ld}HP": [f for f in FILE_IDS[int(ld)] if f in missing] for ld in loads}
        by_load = {k: v for k, v in by_load.items() if v}
        raise FileNotFoundError(
            f"CWRU: no variable X<id>_DE_time found for file ids {by_load} under {root}. Point paths.cwru "
            f"to the folder(s) containing them, or restrict loads, e.g. data.loads=[3]. "
            f"Run scripts/find_cwru.py to see what is available.")
    return recs
