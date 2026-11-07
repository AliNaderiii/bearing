"""IMS (NASA / Univ. Cincinnati) run-to-failure bearing data.

Layout: <root>/1st_test/*, <root>/2nd_test/*, <root>/3rd_test/* (or 4th_test/txt/*).
Each file: 1-second snapshot, 20480 rows @ 20 kHz, whitespace separated
(8 columns in test 1 = 2 per bearing, 4 columns in tests 2/3).

Classes (failure modes at end of test):
  Normal  : early-life snapshots of the same failing channels (before the onset)
  Inner   : test 1, bearing 3          Roller : test 1, bearing 4
  Outer   : test 2, bearing 1  and  test 3, bearing 3
Fault onset is DATA-DRIVEN (not hand-picked): baseline statistics of RMS and
kurtosis on early files, onset = first of ``k`` consecutive files exceeding
mean + n_sigma * std. The result is cached (JSON) and reported.

Limitation (documented in PROTOCOLS.md): each fault class comes from one or two
physical bearings, so IMS measures *temporal* generalisation, not cross-unit.
"""
from __future__ import annotations

import glob
import json
import os
import re
from functools import partial
from typing import Dict, List, Optional

import numpy as np

from ..recording import Recording

FS = 20000.0
# (test folder candidates, bearing, column, label name)
STREAMS = [
    (("1st_test",), 3, 4, "Inner"),
    (("1st_test",), 4, 6, "Roller"),
    (("2nd_test",), 1, 0, "Outer"),
    (("3rd_test", "4th_test"), 3, 2, "Outer"),
]
CLASSES = ["Normal", "Inner", "Roller", "Outer"]


def _read_col(path: str, col: int) -> np.ndarray:
    try:
        import pandas as pd
        a = pd.read_csv(path, sep=r"\s+", header=None, usecols=[col], engine="c").to_numpy(np.float32).ravel()
    except Exception:
        a = np.loadtxt(path, usecols=[col], dtype=np.float32)
    return a[None, :]


_TS = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$")


def _find_files(root: str, folders) -> List[str]:
    """Find the (possibly nested, e.g. 4th_test/txt or 1st_test/1st_test) folder holding the
    timestamp-named snapshot files such as 2003.10.22.12.06.24. Read-only."""
    for f in folders:
        base = os.path.join(root, f)
        if not os.path.isdir(base):
            continue
        best: List[str] = []
        for dirpath, _dirs, filenames in os.walk(base):
            if os.path.relpath(dirpath, base).count(os.sep) > 3:
                continue
            ts = sorted(os.path.join(dirpath, n) for n in filenames if _TS.match(n))
            if len(ts) > len(best):
                best = ts
        if best:
            return best
    return []


def _kurt(x):
    x = x - x.mean()
    s2 = (x ** 2).mean() + 1e-12
    return float((x ** 4).mean() / s2 ** 2)


def detect_onset(files: List[str], col: int, cfg: dict, cache_json: Optional[str] = None) -> Dict:
    if cache_json and os.path.exists(cache_json):
        with open(cache_json) as f:
            return json.load(f)
    rms, kur = [], []
    for p in files:
        x = _read_col(p, col)[0]
        rms.append(float(np.sqrt((x ** 2).mean())))
        kur.append(_kurt(x))
    rms, kur = np.array(rms), np.array(kur)
    n = len(files)
    s0 = int(n * cfg.get("skip_start_frac", 0.02))
    b1 = max(s0 + 10, int(n * cfg.get("baseline_frac", 0.3)))
    ns, k = float(cfg.get("n_sigma", 3.0)), int(cfg.get("k_consecutive", 5))
    th_r = rms[s0:b1].mean() + ns * rms[s0:b1].std()
    th_k = kur[s0:b1].mean() + ns * kur[s0:b1].std()
    alarm = (rms > th_r) | (kur > th_k)
    onset = n - 1
    run = 0
    for i in range(b1, n):
        run = run + 1 if alarm[i] else 0
        if run >= k:
            onset = i - k + 1
            break
    res = {"n_files": n, "baseline": [s0, b1], "onset": int(onset), "thr_rms": float(th_r),
           "thr_kurt": float(th_k), "rms": rms.tolist(), "kurt": kur.tolist()}
    if cache_json:
        os.makedirs(os.path.dirname(cache_json), exist_ok=True)
        with open(cache_json, "w") as f:
            json.dump(res, f)
    return res


def _even(lst, m):
    if not m or len(lst) <= m:
        return lst
    pick = np.linspace(0, len(lst) - 1, int(m)).round().astype(int)
    return [lst[i] for i in pick]


def build(cfg: dict) -> List[Recording]:
    root = cfg["root"]
    ocfg = cfg.get("onset", {})
    margin = int(ocfg.get("margin_files", 20))
    mx = cfg.get("max_files_per_stream")
    cache_dir = cfg.get("cache_dir", "cache")
    recs: List[Recording] = []
    report = {}
    for folders, bearing, col, lname in STREAMS:
        files = _find_files(root, folders)
        if not files:
            if cfg.get("strict", True):
                raise FileNotFoundError(f"IMS folder {folders} not found under {root}")
            continue
        tag = f"{folders[0]}_B{bearing}"
        on = detect_onset(files, col, ocfg, os.path.join(cache_dir, "ims", f"onset_{tag}.json"))
        onset = on["onset"]
        s0, b1 = on["baseline"]
        healthy = list(range(s0, max(s0 + 1, min(b1, onset - margin))))
        faulty = list(range(onset, len(files)))
        report[tag] = {"onset_file": os.path.basename(files[onset]), "onset_idx": onset,
                       "n_files": len(files), "n_healthy": len(healthy), "n_faulty": len(faulty)}
        for lbl_name, idxs in (("Normal", healthy), (lname, faulty)):
            for i in _even(idxs, mx):
                p = files[i]
                recs.append(Recording(
                    rid=f"ims_{tag}_{os.path.basename(p)}", dataset="ims", label=CLASSES.index(lbl_name),
                    label_name=lbl_name, group=tag, condition=tag, fs=FS, loader=partial(_read_col, p, col),
                    channels=[f"B{bearing}"], time_order=float(i), meta={"path": p, "file_idx": i},
                ))
    cfg["_onset_report"] = report
    return recs
