#!/usr/bin/env python
"""Detector-law study (no training): which non-linearity after a band-pass filter keeps the Gaussian-noise
benefit of the square law AND survives impulsive noise?

Fixed band-pass FIR (15 taps, 2.4-4.2 kHz, as in Table 2 / Fig. 2) followed by a detector g(.):
    linear      y
    relu        max(y, 0)                     (linear-law detector after a rectifier)
    abs         |y|                           (linear law)
    square      y^2                           (square law = EQC product/power term, Prop. 2)
    clip_sq     min(y^2, c s^2)               (clipped square law)
    tanh_sq     c s^2 tanh(y^2 / (c s^2))     (smoothly saturated square law)
    log_sq      log(1 + y^2 / s^2)            (log-envelope-type compression)
    hilbert_sq  |y + j H{y}|^2                (classical squared envelope, the SES detector)
    tkeo        y(n)^2 - y(n-1) y(n+1)        (Teager-Kaiser energy operator; a 3-tap 2nd-order Volterra kernel)
    bn_sigmoid  sigmoid(relu((y^2 - mean)/std))   (ClassBD-like; statistics of the NOISY window)
s^2 is a robust noise-level estimate of the NOISY window itself, s^2 = median(y^2)/0.4549 (median of chi^2_1),
i.e. no clean-signal information is used; c = 9 (clip at 3 s in amplitude).

Noise (same power at each SNR, relative to each window): Gaussian, impulsive (model of phyq/data/noise.py,
kurtosis ~250) and the simulator's random impacts (phyq.sim.bearing_sim.add_impulsive, on top of Gaussian).
Metrics (mean over outer / inner / ball faults, 200 windows each, fixed speed):
    prominence  median fault-frequency prominence of g(y) (dB; scripts/sim_mechanism.fault_prominence)
    distortion  power of g(noisy)-g(clean) relative to g(clean) (dB), pooled
    AUC         separability fault vs. normal windows using the prominence at the class's fault frequency

    python scripts/sim_detector_laws.py [--n 200] [--fig <dir>]  -> results/sim_mechanism/detector_laws.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import firwin, hilbert, lfilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phyq.sim.bearing_sim import SimConfig, add_impulsive, simulate_one  # noqa: E402

_spec = importlib.util.spec_from_file_location("sim_mechanism", ROOT / "scripts" / "sim_mechanism.py")
sm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sm)

C_SAT = 9.0
MED_CHI2 = 0.4549


def _s2(y):
    return np.median(y ** 2, axis=-1, keepdims=True) / MED_CHI2 + 1e-12


def _bn_sigmoid(y):
    q = y ** 2
    z = (q - q.mean(-1, keepdims=True)) / (q.std(-1, keepdims=True) + 1e-12)
    return 1.0 / (1.0 + np.exp(-np.maximum(z, 0.0)))


DETECTORS = {
    "linear": lambda y: y,
    "relu": lambda y: np.maximum(y, 0.0),
    "abs": np.abs,
    "square": lambda y: y ** 2,
    "clip_sq": lambda y: np.minimum(y ** 2, C_SAT * _s2(y)),
    "tanh_sq": lambda y: C_SAT * _s2(y) * np.tanh(y ** 2 / (C_SAT * _s2(y))),
    "log_sq": lambda y: np.log1p(y ** 2 / _s2(y)),
    "bn_sigmoid": _bn_sigmoid,
    # v10.5 (reviewer request): classical demodulators for reference
    "hilbert_sq": lambda y: np.abs(hilbert(y, axis=-1)) ** 2,                    # squared envelope (SES)
    "tkeo": lambda y: y ** 2 - np.roll(y, 1, -1) * np.roll(y, -1, -1),          # Teager-Kaiser energy operator
}
LABELS = {"linear": "none (linear)", "relu": "ReLU / |·| (linear law)", "abs": "|·| (linear law)", "square": "square law",
          "clip_sq": "clipped square", "tanh_sq": "tanh-saturated square", "log_sq": "log square",
          "bn_sigmoid": "BN-ReLU-sigmoid (window stats)", "hilbert_sq": "squared Hilbert envelope",
          "tkeo": "Teager-Kaiser energy"}


def make_noise(x, kind, snr, rng):
    p = (x ** 2).mean(-1, keepdims=True)
    if kind == "gaussian":
        n = rng.standard_normal(x.shape)
    elif kind == "impulsive":
        n = 0.3 * rng.standard_normal(x.shape) + (rng.random(x.shape) < 0.01) * rng.standard_normal(x.shape) * 10
    else:
        raise ValueError(kind)
    return n / np.sqrt((n ** 2).mean(-1, keepdims=True)) * np.sqrt(p / 10 ** (snr / 10))


def noisy(x, kind, snr, rng):
    if kind == "impacts":     # simulator interference (non-fault impacts) + Gaussian background at `snr`
        return add_impulsive(x[:, None, :].astype(np.float32), rng)[:, 0] + make_noise(x, "gaussian", snr, rng)
    return x + make_noise(x, kind, snr, rng)


def auc(pos, neg):
    """Mann-Whitney AUC."""
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort() + 1.0
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--fig", default=None, help="directory for the figure (PDF/PNG)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    cfg = SimConfig()
    data = {}
    for c in ("normal", "outer", "inner", "ball"):
        xs, ff = [], None
        for _ in range(a.n):
            x, info = simulate_one(c, rng, cfg, rpm=1797.0, return_info=True)
            xs.append(x)
            ff = info.get("f_fault")
        data[c] = (np.stack(xs).astype(np.float64), ff)
    ffs = {c: data[c][1] for c in ("outer", "inner", "ball")}
    h = firwin(15, [2_400, 4_200], pass_zero=False, fs=sm.FS)
    rows = []
    for kind in ("gaussian", "impulsive", "impacts"):
        for snr in (6, 0, -6, -10):
            nrng = np.random.default_rng(1000 + snr + {"gaussian": 0, "impulsive": 100, "impacts": 200}[kind])
            Y0 = {c: lfilter(h, 1.0, data[c][0], axis=-1) for c in data}
            Y1 = {c: lfilter(h, 1.0, noisy(data[c][0], kind, snr, nrng), axis=-1) for c in data}
            for d, g in DETECTORS.items():
                prom, dist_n, dist_d, aucs = [], 0.0, 0.0, []
                gn = g(Y1["normal"])
                for c, ff in ffs.items():
                    o0, o1 = g(Y0[c]), g(Y1[c])
                    o0c, o1c = o0 - o0.mean(-1, keepdims=True), o1 - o1.mean(-1, keepdims=True)
                    dist_n += float(((o1c - o0c) ** 2).sum())
                    dist_d += float((o0c ** 2).sum())
                    pf = sm.fault_prominence(o1, sm.FS, ff)
                    pn = sm.fault_prominence(gn, sm.FS, ff)
                    prom.append(float(np.median(pf)))
                    aucs.append(auc(pf, pn))
                rows.append({"noise": kind, "snr": snr, "detector": d, "prominence_db": round(float(np.mean(prom)), 2),
                             "distortion_db": round(10 * np.log10(dist_n / dist_d), 2),
                             "auc": round(float(np.mean(aucs)), 3)})
                r = rows[-1]
                print(f"{kind:9s} {snr:4d} dB | {d:10s} | prom {r['prominence_db']:6.1f} | dist {r['distortion_db']:6.1f}"
                      f" | AUC {r['auc']:.3f}", flush=True)
    out = ROOT / "results" / "sim_mechanism"
    out.mkdir(parents=True, exist_ok=True)
    (out / "detector_laws.json").write_text(json.dumps({"c_sat": C_SAT, "n_per_class": a.n, "rows": rows}, indent=1))
    if a.fig:
        figure(rows, Path(a.fig))


def figure(rows, outdir: Path):
    import glob
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    d = os.environ.get("PHYQ_FONT_DIR")
    if d:
        for f in glob.glob(os.path.join(d, "**", "*.ttf"), recursive=True):
            font_manager.fontManager.addfont(f)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                         "axes.linewidth": 0.6})
    show = [("relu", "#2E9E5B", "-", "s"), ("square", "#D1495B", "-", "D"),
            ("clip_sq", "#6C4FA3", "--", "v"), ("log_sq", "#E69F00", "--", "^"), ("bn_sigmoid", "#00798C", ":", "P")]
    noises = [("gaussian", "Gaussian noise"), ("impulsive", "impulsive noise"), ("impacts", "random impacts + Gaussian")]
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.2), sharex=True)
    snrs = [6, 0, -6, -10]
    for j, (nk, nt) in enumerate(noises):
        for i, (met, yl) in enumerate((("prominence_db", "fault-frequency prominence (dB)"), ("auc", "AUC fault vs normal"))):
            ax = axs[i, j]
            for dk, col, ls, mk in show:
                v = [next(r[met] for r in rows if r["noise"] == nk and r["snr"] == s and r["detector"] == dk) for s in snrs]
                ax.plot(snrs, v, ls=ls, marker=mk, color=col, lw=1.1, ms=3.2, label=LABELS[dk])
            ax.invert_xaxis()
            ax.set_xticks(snrs)
            if i == 0:
                ax.set_title(nt, fontsize=7.5, fontweight="bold")
            else:
                ax.set_xlabel("SNR (dB)")
                ax.set_ylim(0.45, 1.02)
            if j == 0:
                ax.set_ylabel(yl)
            ax.grid(alpha=0.25, lw=0.4)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=6.5, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "Figure_S2_detector_laws.pdf")
    fig.savefig(outdir / "Figure_S2_detector_laws.png", dpi=300)


if __name__ == "__main__":
    main()
