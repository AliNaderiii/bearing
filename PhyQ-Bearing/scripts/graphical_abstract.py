#!/usr/bin/env python
"""Graphical abstract (MSSP: >= 1328 px wide, readable at 13 cm). Data: simulator + sandbox sim results."""
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

out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "sim_mechanism" / "figures"
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
rng = np.random.default_rng(1)
x0, info = simulate_one("outer", rng, SimConfig(), rpm=1797.0, return_info=True)
x = x0 + rng.standard_normal(len(x0)) * x0.std()          # 0 dB Gaussian noise
fs = 12000.0
t = np.arange(len(x)) / fs
h = firwin(15, [2400, 4200], pass_zero=False, fs=fs)
y = lfilter(h, 1, x)
f = np.fft.rfftfreq(len(x), 1 / fs)
sel = (f > 5) & (f < 600)

def spec(v):
    v = v - v.mean()
    P = np.abs(np.fft.rfft(v * np.hanning(len(v)))) ** 2
    return 10 * np.log10(P / np.median(P[sel]))

fig = plt.figure(figsize=(13.3, 5.0), dpi=110)
gs = fig.add_gridspec(2, 4, width_ratios=[1.2, 1, 1, 1.1], hspace=0.55, wspace=0.45)
ax = fig.add_subplot(gs[0, 0]); ax.plot(t[:1200] * 1e3, x0[:1200], lw=0.5, color="k")
ax.set_title("Bearing signal: impacts excite a resonance"); ax.set_xlabel("Time (ms)"); ax.set_yticks([])
ax = fig.add_subplot(gs[1, 0]); ax.plot(f[sel], spec(x)[sel], lw=0.6, color="0.4")
ax.set_title("Raw spectrum (0 dB noise)"); ax.set_xlabel("Frequency (Hz)")
for k in (1, 2, 3):
    ax.axvline(k * info["f_fault"], color="tab:red", ls="--", lw=0.8, zorder=0)
ax = fig.add_subplot(gs[:, 1]); ax.axis("off")
ax.text(0.5, 0.92, "Quadratic convolution", ha="center", weight="bold", fontsize=11)
ax.text(0.5, 0.80, r"$(w_r*x)(w_g*x) + w_b*(x^2)$", ha="center", fontsize=11)
for i, (txt, c) in enumerate([("band-pass", "#4c72b0"), ("square law", "#c44e52"), ("low-pass (next layers)", "#55a868")]):
    ax.add_patch(plt.Rectangle((0.1, 0.55 - 0.2 * i), 0.8, 0.13, color=c, alpha=0.25, transform=ax.transAxes))
    ax.text(0.5, 0.615 - 0.2 * i, txt, ha="center", va="center", fontsize=10)
ax.text(0.5, 0.02, "= learnable squared-envelope detector\n(linear conv: cannot; ReLU: linear law)",
        ha="center", fontsize=9, style="italic")
ax = fig.add_subplot(gs[0, 2]); ax.plot(f[sel], spec(y)[sel], lw=0.6, color="0.4"); ax.set_title("Linear (band-pass) output")
for k in (1, 2, 3):
    ax.axvline(k * info["f_fault"], color="tab:red", ls="--", lw=0.8, zorder=0)
ax = fig.add_subplot(gs[1, 2]); ax.plot(f[sel], spec(y ** 2)[sel], lw=0.6, color="k"); ax.set_title("Square-law output")
ax.set_xlabel("Frequency (Hz)")
for k in (1, 2, 3):
    ax.axvline(k * info["f_fault"], color="tab:red", ls="--", lw=0.8, zorder=0)
ax = fig.add_subplot(gs[:, 3])
acc = {}
for name in ("quad", "noquad"):
    p = ROOT / "results" / "sim_mechanism" / f"{name}_seed0" / "metrics.json"
    if p.exists():
        m = json.loads(p.read_text()); acc[name] = [100 * m["white-10"]["acc"], 100 * m["impulsive"]["acc"]]
if acc:
    xs = np.arange(2)
    ax.bar(xs - 0.2, acc.get("quad", [0, 0]), 0.4, color="k", label="quadratic (square law)")
    ax.bar(xs + 0.2, acc.get("noquad", [0, 0]), 0.4, color="0.7", label="linear twin (linear law)")
    ax.set_xticks(xs, ["Gaussian\n-10 dB", "Impulsive"]); ax.set_ylabel("Accuracy (%), simulator, 1 seed"); ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_title("Trade-off predicted by theory:\noutput noise grows with kurtosis")
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "Graphical_abstract.png", dpi=160, bbox_inches="tight")
fig.savefig(out / "Graphical_abstract.pdf", bbox_inches="tight")
from PIL import Image
print(Image.open(out / "Graphical_abstract.png").size)
