#!/usr/bin/env python
"""Figures of the mechanism study (simulated signals with known fault frequency).

  python scripts/sim_figures.py --out <dir>      (default: results/sim_mechanism/figures)

Figure_2_detectors.pdf : spectra of the output of a fixed band-pass kernel followed by
                         no detector / ReLU / square law, outer-race fault, 0 dB Gaussian vs impulsive noise.
Figure_3_prominence.pdf: first-layer fault-frequency prominence of trained models (needs
                         results/sim_mechanism/prominence.json from `sim_mechanism.py analyze`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.signal import firwin, lfilter  # noqa: E402

from phyq.sim.bearing_sim import SimConfig, simulate_one  # noqa: E402

FS = 12_000.0
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "ps.fonttype": 42})


def _noise(x, kind, snr, rng):
    if kind == "white":
        n = rng.standard_normal(x.shape)
    else:
        n = 0.3 * rng.standard_normal(x.shape) + (rng.random(x.shape) < 0.01) * rng.standard_normal(x.shape) * 10
    p = (x ** 2).mean(-1, keepdims=True)
    return n / np.sqrt((n ** 2).mean(-1, keepdims=True)) * np.sqrt(p / 10 ** (snr / 10))


def fig_detectors(out: Path, n_avg: int = 40, snr: float = 0.0):
    rng = np.random.default_rng(3)
    cfg = SimConfig()
    xs, ff = [], None
    for _ in range(n_avg):
        x, info = simulate_one("outer", rng, cfg, rpm=1797.0, return_info=True)
        xs.append(x); ff = info["f_fault"]
    X = np.stack(xs)
    h = firwin(15, [2_400, 4_200], pass_zero=False, fs=FS)
    dets = [("Linear (no detector)", lambda y: y), ("ReLU (linear law)", lambda y: np.maximum(y, 0)),
            ("Square law", lambda y: y ** 2)]
    f = np.fft.rfftfreq(X.shape[-1], 1 / FS)
    sel = f <= 600
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.6), sharex=True)
    for r, kind in enumerate(("white", "impulsive")):
        Xn = X + _noise(X, kind, snr, rng)
        Y = lfilter(h, 1.0, Xn, axis=-1)
        for c, (name, fn) in enumerate(dets):
            o = fn(Y); o = o - o.mean(-1, keepdims=True)
            P = (np.abs(np.fft.rfft(o * np.hanning(o.shape[-1]), axis=-1)) ** 2).mean(0)
            P = 10 * np.log10(P / np.median(P[sel & (f > 15)]))
            ax = axes[r, c]
            ax.plot(f[sel], P[sel], lw=0.6, color="k")
            for k in (1, 2, 3):
                ax.axvline(k * ff, color="tab:red", lw=0.5, ls="--", alpha=0.7)
            ax.set_ylim(-10, 25)
            if r == 0:
                ax.set_title(name)
            if c == 0:
                ax.set_ylabel(("Gaussian" if kind == "white" else "Impulsive") + f", {snr:g} dB\nPSD (dB re median)")
            if r == 1:
                ax.set_xlabel("Frequency (Hz)")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "Figure_2_detectors.pdf"); fig.savefig(out / "Figure_2_detectors.png", dpi=300)
    plt.close(fig)


def fig_prominence(out: Path):
    p = ROOT / "results" / "sim_mechanism" / "prominence.json"
    if not p.exists():
        print("prominence.json missing -> run sim_mechanism.py analyze first"); return
    rows = json.loads(p.read_text())
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["snr"], r["model"], r["layer"])].append(r["top10_db"] if r["layer"] != "input" else r["mean_db"])
    bars = [("raw input", ("raw_input", "input")),
            ("linear twin, pre-act.", ("noquad", "pre")), ("linear twin, post-act.", ("noquad", "post")),
            ("quadratic, pre-BN", ("quad", "pre")), ("quad: product term", ("quad", "product_rg")),
            ("quad: power term", ("quad", "power_b")), ("quadratic, post-act.", ("quad", "post")),
            ("quadratic (random init)", ("quad_init", "pre"))]
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    w = 0.38
    for i, snr in enumerate(("clean", 0)):
        vals = [np.nanmean(agg.get((snr, m, l), [np.nan])) for _, (m, l) in bars]
        ax.bar(np.arange(len(bars)) + (i - 0.5) * w, vals, w, label="clean" if snr == "clean" else "0 dB Gaussian",
               color=["0.3", "0.7"][i])
    ax.set_xticks(np.arange(len(bars)), [b[0] for b in bars], rotation=20, ha="right")
    ax.set_ylabel("Fault-frequency\nprominence (dB)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "Figure_3_prominence.pdf"); fig.savefig(out / "Figure_3_prominence.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "sim_mechanism" / "figures"))
    a = ap.parse_args()
    fig_detectors(Path(a.out))
    fig_prominence(Path(a.out))
