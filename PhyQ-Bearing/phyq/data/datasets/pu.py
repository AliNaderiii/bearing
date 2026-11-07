"""Paderborn University (PU / KAt) bearing dataset.

Expected layout (as distributed):  <root>/<BEARING>/<COND>_<BEARING>_<k>.mat
e.g. ``K001/N15_M07_F10_K001_1.mat``. Each file holds a struct named after the
file stem; ``Y`` contains named channels, the vibration is ``vibration_1``
(64 kHz). Motor currents (``phase_current_1/2``, 64 kHz) can be added.

The physical unit (``group``) is the bearing code, so group-wise splits are
truly bearing-disjoint (the legacy pipeline split windows/files at random).
"""
from __future__ import annotations

import glob
import os
import re
from functools import partial
from typing import Dict, List

import numpy as np
import scipy.io as sio

from ..recording import Recording

FS = 64000.0
HEALTHY = ["K001", "K002", "K003", "K004", "K005", "K006"]
ARTIFICIAL = ["KA01", "KA03", "KA05", "KA06", "KA07", "KA08", "KA09", "KI01", "KI03", "KI05", "KI07", "KI08"]
REAL = ["KA04", "KA15", "KA16", "KA22", "KA30", "KI04", "KI14", "KI16", "KI17", "KI18", "KI21",
        "KB23", "KB24", "KB27"]
CONDITIONS = ["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]

DEFAULT_CLASSES = {  # 3-class problem used by most PU papers
    "Healthy": ["K001", "K002", "K003", "K004", "K005"],
    "OR": ["KA04", "KA15", "KA16", "KA22", "KA30"],
    "IR": ["KI04", "KI14", "KI16", "KI18", "KI21"],
}


def _load(path: str, channels: List[str]) -> np.ndarray:
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    stem = os.path.splitext(os.path.basename(path))[0]
    keys = [k for k in m if not k.startswith("__")]
    key = stem if stem in m else keys[0]
    s = m[key]
    ys = np.atleast_1d(s.Y)
    out = []
    for ch in channels:
        hit = [y for y in ys if str(getattr(y, "Name", "")) == ch]
        if not hit:
            raise KeyError(f"{path}: channel {ch} not found; have {[str(y.Name) for y in ys]}")
        out.append(np.asarray(hit[0].Data, dtype=np.float32).ravel())
    n = min(len(o) for o in out)
    return np.stack([o[:n] for o in out], 0)


def build(cfg: dict) -> List[Recording]:
    root = cfg["root"]
    classes: Dict[str, List[str]] = cfg.get("classes", DEFAULT_CLASSES)
    conds = cfg.get("conditions", CONDITIONS)
    channels = cfg.get("channels", ["vibration_1"])
    max_files = cfg.get("max_files_per_condition")
    b2label = {}
    for li, (name, bs) in enumerate(classes.items()):
        for b in bs:
            b2label[b] = (li, name)
    recs: List[Recording] = []
    for b, (li, name) in sorted(b2label.items()):
        files = sorted(glob.glob(os.path.join(root, "**", f"*_{b}_*.mat"), recursive=True))
        if not files:
            if b in (cfg.get("optional_bearings") or []):
                print(f"[pu] optional bearing {b} not found -> skipped")
                continue
            if cfg.get("strict", True):
                raise FileNotFoundError(f"PU bearing {b} not found under {root}")
            continue
        per_cond: Dict[str, list] = {}
        for f in files:
            mm = re.match(r"(N\d+_M\d+_F\d+)_" + b + r"_(\d+)\.mat$", os.path.basename(f))
            if not mm or mm.group(1) not in conds:
                continue
            per_cond.setdefault(mm.group(1), []).append((int(mm.group(2)), f))
        for c, lst in per_cond.items():
            lst.sort()
            if max_files:
                lst = lst[: int(max_files)]
            for k, f in lst:
                recs.append(Recording(
                    rid=f"pu_{c}_{b}_{k}", dataset="pu", label=li, label_name=name, group=b,
                    condition=c, fs=FS, loader=partial(_load, f, channels), channels=channels,
                    time_order=k, meta={"path": f, "bearing": b,
                                        "damage": "healthy" if b in HEALTHY else
                                        ("artificial" if b in ARTIFICIAL else "real")},
                ))
    return recs
