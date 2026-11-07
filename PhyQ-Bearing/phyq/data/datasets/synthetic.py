"""Synthetic bearing-like signals for smoke tests (no download needed).

Each class = impulse train at a class-specific characteristic frequency exciting a
resonance, plus shaft harmonics and noise. Several 'units' per class allow
group-wise protocols to be exercised.
"""
from __future__ import annotations

from functools import partial
from typing import List

import numpy as np

from ..recording import Recording


def _gen(label: int, unit: int, cond: int, n: int, fs: float, n_ch: int) -> np.ndarray:
    rng = np.random.RandomState(1000 * label + 37 * unit + cond)
    t = np.arange(n) / fs
    shaft = 20 + 5 * cond
    x = 0.3 * np.sin(2 * np.pi * shaft * t) + 0.1 * np.sin(2 * np.pi * 2 * shaft * t)
    if label > 0:
        fc = shaft * (2.5 + 1.3 * label)
        period = max(int(fs / fc), 1)
        imp = np.zeros(n)
        imp[::period] = 1.0 + 0.1 * rng.randn(len(imp[::period]))
        fr = 2000 + 300 * unit
        k = np.arange(int(0.004 * fs)) / fs
        h = np.exp(-800 * k) * np.sin(2 * np.pi * fr * k)
        x = x + 0.8 * np.convolve(imp, h, mode="same")
    out = [x + 0.2 * rng.randn(n) for _ in range(n_ch)]
    return np.stack(out, 0).astype(np.float32)


def build(cfg: dict) -> List[Recording]:
    n_cls = int(cfg.get("n_classes", 4))
    n_units = int(cfg.get("n_units", 3))
    n_cond = int(cfg.get("n_conditions", 2))
    n = int(cfg.get("length", 60000))
    fs = float(cfg.get("fs", 12000))
    n_ch = int(cfg.get("n_channels", 1))
    recs = []
    for c in range(n_cls):
        for u in range(n_units):
            for k in range(n_cond):
                recs.append(Recording(
                    rid=f"syn_c{c}_u{u}_k{k}", dataset="synthetic", label=c, label_name=f"C{c}",
                    group=f"C{c}U{u}", condition=f"k{k}", fs=fs,
                    loader=partial(_gen, c, u, k, n, fs, n_ch), channels=[f"ch{i}" for i in range(n_ch)],
                ))
    return recs
