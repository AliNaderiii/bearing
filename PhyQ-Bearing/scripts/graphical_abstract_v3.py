#!/usr/bin/env python
"""Graphical abstract v3 (eight-panel flow, layout proposed by the authors), fully data-derived and vector.

  * panels 1, 3, 4, 6: phyq.sim.bearing_sim with the generator, seed, noise and FIR filter of Fig. 2
    (so the printed peak values are identical to Fig. 2);
  * panel 8: user GPU results (review/user_results), absolute SNR, test sets, mean over tasks/seeds;
  * panels 2, 5, 7: schematic drawings (no data).
Native size 14 cm x 8 cm (readable at the 13 cm width required by Elsevier; PNG 600 dpi > 1328 px).

    PHYQ_FONT_DIR=/path/to/fonts python scripts/graphical_abstract_v3.py --out <dir>
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

ASPECT = 14 / 8          # figure width / height -> keeps circles round in figure coordinates


def Circle(xy, r, **kw):
    return Ellipse(xy, 2 * r, 2 * r * ASPECT, **kw)  # noqa: E402

NAVY, RED, TEAL, GREY, INK, AMBER, PURPLE = "#1F4E79", "#D1495B", "#00798C", "#5D6D7E", "#1F2328", "#D98E1F", "#6C4FA3"
BLUE, ORANGE = "#2C7FB8", "#E69F00"
Q_BG, A_BG, S_BG, N_BG, F_BG = "#FBE3E6", "#DDF0F3", "#FDF0DC", "#F1F2F4", "#ECE6F5"

ROWS = ["SDUST", "PU 5-fold", "PU art→real", "IMS", "CWRU", "NREL"]
GAUSS10 = {"SDUST": (66.28, 59.1), "PU 5-fold": (60.4, 43.9), "PU art→real": (47.0, 44.1),
           "IMS": (64.49, 71.74), "CWRU": (52.25, 64.72), "NREL": (100.0, 98.0)}          # (PhyQ-TransNet, ClassBD)
IMP6 = {"SDUST": (42.7, 32.9), "PU 5-fold": (38.5, 35.1), "PU art→real": (31.7, 52.1),
        "IMS": (35.9, 41.0), "CWRU": (12.1, 40.5), "NREL": (50.0, 99.5)}


def setup():
    d = os.environ.get("PHYQ_FONT_DIR")
    if d and os.path.isdir(d):
        for f in glob.glob(os.path.join(d, "**", "*.ttf"), recursive=True):
            font_manager.fontManager.addfont(f)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
        "font.size": 5.6, "axes.linewidth": 0.45, "xtick.major.width": 0.45, "ytick.major.width": 0.45,
        "xtick.major.size": 1.6, "ytick.major.size": 1.6, "xtick.major.pad": 1.2, "ytick.major.pad": 1.2,
        "xtick.labelsize": 4.9, "ytick.labelsize": 4.9, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def simulate():
    """Fig. 2 computation (same rng order): Gaussian and impulsive noise at 0 dB, linear / square-law outputs."""
    from scipy.signal import firwin, freqz, lfilter
    from phyq.sim.bearing_sim import SimConfig, simulate_one
    fs, snr, n_avg = 12_000.0, 0.0, 40
    rng = np.random.default_rng(3)
    cfg = SimConfig()
    xs, ff = [], None
    for _ in range(n_avg):
        x, info = simulate_one("outer", rng, cfg, rpm=1797.0, return_info=True)
        xs.append(x)
        ff = info["f_fault"]
    X = np.stack(xs)

    def noise(x, kind):
        if kind == "white":
            n = rng.standard_normal(x.shape)
        else:
            n = 0.3 * rng.standard_normal(x.shape) + (rng.random(x.shape) < 0.01) * rng.standard_normal(x.shape) * 10
        p = (x ** 2).mean(-1, keepdims=True)
        return n / np.sqrt((n ** 2).mean(-1, keepdims=True)) * np.sqrt(p / 10 ** (snr / 10))

    h = firwin(15, [2_400, 4_200], pass_zero=False, fs=fs)
    f = np.fft.rfftfreq(X.shape[-1], 1 / fs)
    sel = f <= 600
    out = dict(fs=fs, f=f, ff=ff, clean=X[0])
    for kind in ("white", "impulsive"):
        Xn = X + noise(X, kind)
        out["noisy_" + kind] = Xn[0]
        Y = lfilter(h, 1.0, Xn, axis=-1)
        for name, o in (("linear", Y), ("square", Y ** 2)):
            o = o - o.mean(-1, keepdims=True)
            P = (np.abs(np.fft.rfft(o * np.hanning(o.shape[-1]), axis=-1)) ** 2).mean(0)
            P = 10 * np.log10(P / np.median(P[sel & (f > 15)]))
            out[f"{name}_{kind}"] = P
            out[f"{name}_{kind}_peak"] = P[np.abs(f - ff) < 4].max()
    w, H = freqz(h, worN=512, fs=fs)
    out["fir_f"], out["fir_H"] = w, np.abs(H)
    X0 = np.abs(np.fft.rfft(X * np.hanning(X.shape[-1]), axis=-1)) ** 2
    out["sig_psd"] = X0.mean(0)
    return out


class Canvas:
    def __init__(self, fig):
        self.fig = fig
        self.bg = fig.add_axes([0, 0, 1, 1], zorder=-10)
        self.bg.axis("off")
        self.bg.set_xlim(0, 1)
        self.bg.set_ylim(0, 1)

    def box(self, x, y, w, h, fc="white", ec=NAVY, lw=0.6, r=0.01, z=0):
        self.bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                                         lw=lw, zorder=z))

    def panel(self, n, x, y, w, h, title, col=NAVY, fc="white"):
        self.box(x, y, w, h, fc=fc, ec=col, lw=0.7)
        cx, cy = x + 0.016, y + h - 0.037
        self.bg.add_patch(Circle((cx, cy), 0.0105, fc=col, ec="none", zorder=3))
        self.bg.text(cx, cy, str(n), color="white", fontsize=5.6, fontweight="bold", ha="center", va="center",
                     zorder=4)
        self.bg.text(cx + 0.015, cy, title, color=col, fontsize=5.15, fontweight="bold", ha="left", va="center",
                     linespacing=1.05)

    def arrow(self, p0, p1, col=NAVY, lw=0.8, label=None, lpos=None, conn="arc3"):
        self.bg.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=6, lw=lw, color=col,
                                          connectionstyle=conn, zorder=5, shrinkA=0, shrinkB=0))
        if label:
            lx, ly = lpos if lpos else ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + 0.022)
            self.bg.text(lx, ly, label, color=col, fontsize=4.9, ha="center", va="center", style="italic")

    def text(self, x, y, s, **kw):
        kw.setdefault("fontsize", 5.2)
        kw.setdefault("ha", "center")
        kw.setdefault("va", "center")
        kw.setdefault("color", INK)
        self.bg.text(x, y, s, **kw)

    def block(self, x, y, w, h, s, fc, ec, fs=5.0, bold=False):
        self.box(x, y, w, h, fc=fc, ec=ec, lw=0.5, r=0.006, z=2)
        self.bg.text(x + w / 2, y + h / 2, s, fontsize=fs, ha="center", va="center", color=INK, zorder=3,
                     fontweight="bold" if bold else "normal", linespacing=1.0)


def mini(fig, rect):
    ax = fig.add_axes(rect)
    ax.patch.set_alpha(0)
    return ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "_paper_figures"))
    ap.add_argument("--name", default="Graphical_abstract")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    setup()
    S = simulate()
    fig = plt.figure(figsize=(14 / 2.54, 8 / 2.54))
    C = Canvas(fig)

    # ------------------------------------------------------------------ title bar
    C.box(0.006, 0.905, 0.988, 0.088, fc=NAVY, ec=NAVY, r=0.012)
    C.text(0.5, 0.962, "PhyQ-TransNet: quadratic convolution as a learnable square-law envelope detector",
           fontsize=7.4, color="white", fontweight="bold")
    C.text(0.5, 0.926, "theory (3 propositions)  •  simulation  •  5 leakage-controlled bearing datasets  •  "
           "strengths and failure mode reported", fontsize=5.2, color="#DCE6F2")

    # ------------------------------------------------------------------ row 1
    y1, h1, w1, g = 0.475, 0.405, 0.176, 0.0265
    xs = [0.006 + i * (w1 + g) for i in range(5)]
    titles = ["Raw vibration", "Quadratic conv. (EQC)", "Band-pass + square law", "Lines appear at $f_o$",
              "Two gated streams"]
    cols = [GREY, RED, RED, RED, TEAL]
    for i, (x, t, c) in enumerate(zip(xs, titles, cols)):
        C.panel(i + 1, x, y1, w1, h1, t, col=c)
        if i < 4:
            C.arrow((x + w1 + 0.003, y1 + h1 / 2), (x + w1 + g - 0.003, y1 + h1 / 2), col=NAVY)
    fs, f, ff = S["fs"], S["f"], S["ff"]

    # (1) waveform
    x = xs[0]
    ax = mini(fig, [x + 0.012, y1 + 0.185, w1 - 0.024, 0.145])
    n = int(0.05 * fs)
    t = np.arange(n) / fs * 1e3
    ax.plot(t, S["noisy_white"][:n], color="0.75", lw=0.3)
    ax.plot(t, S["clean"][:n], color=INK, lw=0.35)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 25, 50])
    ax.set_xlabel("time (ms)", fontsize=4.8, labelpad=0.5)
    C.text(x + w1 / 2, y1 + 0.068, "fault impacts (black) ring a\nkHz resonance, buried in\n0 dB noise (grey)",
           fontsize=5.0, linespacing=1.15)

    # (2) EQC schematic
    x = xs[1]
    bx, bw, bh = x + 0.055, 0.052, 0.04
    rows_y = [y1 + 0.27, y1 + 0.195, y1 + 0.12]
    labels = ["$W_r\\ast x$", "$W_g\\ast x$", "$W_b\\ast(x\\odot x)$"]
    for yy, lab, fc in zip(rows_y, labels, ["#FFFFFF", "#FFFFFF", "#FFFFFF"]):
        C.block(bx, yy, bw + (0.022 if "W_b" in lab else 0), bh, lab, fc, RED, fs=4.9)
        C.arrow((x + 0.012, yy + bh / 2), (bx - 0.002, yy + bh / 2), col=GREY, lw=0.5)
    C.text(x + 0.011, y1 + 0.33, "$x$", fontsize=5.4, ha="left")
    mx, my = x + 0.142, y1 + 0.255
    C.bg.add_patch(Circle((mx, my), 0.012, fc="white", ec=RED, lw=0.6, zorder=3))
    C.text(mx, my, "×", fontsize=6.5, color=RED, zorder=4)
    px, py = x + 0.142, y1 + 0.165
    C.bg.add_patch(Circle((px, py), 0.012, fc="white", ec=RED, lw=0.6, zorder=3))
    C.text(px, py, "+", fontsize=6.5, color=RED, zorder=4)
    C.arrow((bx + bw, rows_y[0] + bh / 2), (mx - 0.009, my + 0.006), col=GREY, lw=0.5)
    C.arrow((bx + bw, rows_y[1] + bh / 2), (mx - 0.009, my - 0.006), col=GREY, lw=0.5)
    C.arrow((mx, my - 0.021), (px, py + 0.021), col=GREY, lw=0.5)
    C.arrow((bx + bw + 0.022, rows_y[2] + bh / 2), (px - 0.012, py), col=GREY, lw=0.5)
    C.text(x + w1 / 2, y1 + 0.06, "bilinear (product) term +\npower term in one layer", fontsize=5.0, linespacing=1.15)

    # (3) band-pass + square law
    x = xs[2]
    ax = mini(fig, [x + 0.016, y1 + 0.175, 0.085, 0.14])
    ax.fill_between(S["fir_f"] / 1e3, 0, S["fir_H"] / S["fir_H"].max(), color=RED, alpha=0.2, lw=0)
    ax.plot(S["fir_f"] / 1e3, S["fir_H"] / S["fir_H"].max(), color=RED, lw=0.6)
    ax.set_xlim(0, 6)
    ax.set_xticks([0, 3, 6])
    ax.set_yticks([])
    ax.set_xlabel("kHz", fontsize=4.8, labelpad=0.5)
    C.arrow((x + 0.106, y1 + 0.245), (x + 0.12, y1 + 0.245), col=GREY, lw=0.6)
    C.block(x + 0.123, y1 + 0.215, 0.044, 0.06, "$(\\cdot)^2$", "#FFF3D6", AMBER, fs=7)
    C.text(x + w1 / 2, y1 + 0.075, "Prop. 2: product term =\nband-pass (resonance) +\n"
           "square-law envelope detector", fontsize=5.0, linespacing=1.15)

    # (4) envelope lines: linear vs square law (Gaussian 0 dB)
    x = xs[3]
    sel = f <= 400
    for k, (key, col, lab) in enumerate((("linear_white", GREY, "linear"), ("square_white", RED, "square law"))):
        ax = mini(fig, [x + 0.02 + k * 0.083, y1 + 0.165, 0.064, 0.14])
        P = S[key]
        ax.fill_between(f[sel], -10, P[sel], color=col, alpha=0.18, lw=0)
        ax.plot(f[sel], P[sel], color=col, lw=0.45)
        for kk in (1, 2, 3):
            ax.axvline(kk * ff, color="0.4", lw=0.3, ls=(0, (2, 2)), zorder=0)
        ax.set_ylim(-10, 24)
        ax.set_xlim(0, 400)
        ax.set_xticks([0, 200, 400] if k == 0 else [200, 400])
        ax.tick_params(axis="x", labelsize=4.5)
        ax.set_yticks([0, 20] if k == 0 else [])
        if k == 1:
            ax.spines["left"].set_visible(False)
        ax.set_title(f"{lab}\n{S[key + '_peak']:+.1f} dB", fontsize=4.8, color=col, pad=1, fontweight="bold",
                     linespacing=1.0)
        ax.set_xlabel("Hz", fontsize=4.8, labelpad=0.5)
    C.text(x + w1 / 2, y1 + 0.075, "Prop. 1: a linear stack cannot\ncreate lines at $f_o$ (0 dB noise,\n"
           "peak re. noise floor)", fontsize=5.0, linespacing=1.15)

    # (5) architecture schematic
    x = xs[4]
    bw5 = w1 - 0.024
    C.block(x + 0.012, y1 + 0.285, bw5, 0.035, "quadratic CNN + QRes", Q_BG, RED, fs=4.7)
    C.block(x + 0.012, y1 + 0.235, 0.098, 0.035, "Transformer", A_BG, TEAL, fs=4.7)
    C.block(x + 0.116, y1 + 0.235, bw5 - 0.104, 0.035, "11 stats\nof raw $x$", S_BG, AMBER, fs=4.3)
    C.arrow((x + 0.061, y1 + 0.285), (x + 0.061, y1 + 0.271), col=GREY, lw=0.5)
    C.block(x + 0.012, y1 + 0.178, bw5, 0.038, "gate: $(1-\\alpha)\\,z_d+\\alpha\\,z_s$", F_BG, PURPLE, fs=4.8)
    C.arrow((x + 0.061, y1 + 0.235), (x + 0.061, y1 + 0.217), col=GREY, lw=0.5)
    C.arrow((x + 0.141, y1 + 0.235), (x + 0.141, y1 + 0.217), col=GREY, lw=0.5)
    C.block(x + 0.045, y1 + 0.128, bw5 - 0.066, 0.032, "classifier", N_BG, GREY, fs=4.7)
    C.arrow((x + w1 / 2, y1 + 0.178), (x + w1 / 2, y1 + 0.161), col=GREY, lw=0.5)
    C.text(x + w1 / 2, y1 + 0.065, "user's original design kept;\nNeural-MED pre-filter off\n(validation ablation)",
           fontsize=5.0, linespacing=1.15)

    # ------------------------------------------------------------------ row 2
    y2, h2 = 0.03, 0.40
    x6, w6 = 0.006, 0.300
    x7, w7 = 0.335, 0.262
    x8, w8 = 0.626, 0.368
    C.panel(6, x6, y2, w6, h2, "Predicted trade-off (Prop. 3)", col=AMBER)
    C.panel(7, x7, y2, w7, h2, "Five datasets, leakage-controlled", col=NAVY)
    C.panel(8, x8, y2, w8, h2, "Accuracy vs ClassBD (strongest baseline)", col=TEAL)
    C.arrow((xs[2] + w1 / 2, y1 - 0.003), (x6 + w6 * 0.55, y2 + h2 + 0.004), col=AMBER, label="predicts",
            lpos=(xs[2] + w1 / 2 - 0.085, y1 - 0.024), conn="angle,angleA=-90,angleB=0,rad=0")
    C.arrow((x6 + w6 + 0.003, y2 + h2 / 2), (x7 - 0.003, y2 + h2 / 2), col=NAVY)
    C.arrow((x7 + w7 + 0.003, y2 + h2 / 2), (x8 - 0.003, y2 + h2 / 2), col=NAVY)

    # (6) square law under Gaussian vs impulsive noise
    for k, (kind, col, lab) in enumerate((("white", BLUE, "Gaussian noise"), ("impulsive", ORANGE, "impulsive noise"))):
        ax = mini(fig, [x6 + 0.025 + k * 0.137, y2 + 0.165, 0.115, 0.135])
        P = S[f"square_{kind}"]
        ax.fill_between(f[sel], -10, P[sel], color=col, alpha=0.2, lw=0)
        ax.plot(f[sel], P[sel], color=col, lw=0.45)
        for kk in (1, 2, 3):
            ax.axvline(kk * ff, color="0.4", lw=0.3, ls=(0, (2, 2)), zorder=0)
        ax.set_ylim(-10, 24)
        ax.set_xlim(0, 400)
        ax.set_xticks([0, 200, 400])
        ax.set_yticks([0, 20] if k == 0 else [])
        if k == 1:
            ax.spines["left"].set_visible(False)
        ax.set_title(f"{lab}, 0 dB\npeak {S[f'square_{kind}_peak']:+.1f} dB", fontsize=4.8, color=col,
                     pad=1, fontweight="bold", linespacing=1.0)
        ax.set_xlabel("Hz", fontsize=4.8, labelpad=0.5)
    C.text(x6 + w6 / 2, y2 + 0.085, "$\\mathrm{Var}=4s^2\\sigma_w^2+4s\\,E[n^3]+(\\kappa_w-1)\\,\\sigma_w^4$",
           fontsize=5.6)
    C.text(x6 + w6 / 2, y2 + 0.035, "Gaussian noise: gain  •  heavy-tailed noise\n"
           "(impulsive, $\\kappa_w$ = 83.8): fragile", fontsize=5.0, linespacing=1.15)

    # (7) datasets and protocol
    items = [("CWRU", "4 loads, chronological"), ("PU", "bearing-disjoint 5-fold; art→real"),
             ("IMS", "run-to-failure, chronological"), ("NREL", "gearbox, healthy vs damaged"),
             ("SDUST", "chronological split")]
    for i, (nm, pr) in enumerate(items):
        yy = y2 + h2 - 0.088 - i * 0.045
        C.block(x7 + 0.014, yy - 0.016, 0.062, 0.032, nm, A_BG if i % 2 == 0 else N_BG, NAVY, fs=4.9, bold=True)
        C.text(x7 + 0.082, yy, pr, fontsize=4.9, ha="left")
    C.text(x7 + w7 / 2, y2 + 0.05, "3 seeds • per-model hyper-parameter search\non validation • 7 baselines • "
           "paired tests,\nHolm-corrected (Wilcoxon / McNemar)", fontsize=4.8, linespacing=1.15)

    # (8) dumbbells
    yv = np.arange(len(ROWS))[::-1]
    for j, (data, ttl, col) in enumerate(((GAUSS10, "Gaussian −10 dB", BLUE), (IMP6, "Impulsive −6 dB", ORANGE))):
        ax = mini(fig, [x8 + 0.07 + j * 0.15, y2 + 0.1, 0.095, 0.21])
        ax.set_facecolor("#F5FAFB")
        ax.patch.set_alpha(1)
        for yi, r in zip(yv, ROWS):
            v = data[r]
            if v is None:
                ax.text(57, yi, "n/a (binary, ceiling)", fontsize=4.3, color="0.5", ha="center", va="center")
                continue
            p, c = v
            win = p > c
            ax.plot([c, p], [yi, yi], color=RED if win else "0.6", lw=1.1, alpha=0.45, solid_capstyle="round")
            ax.scatter([c], [yi], s=6, color=TEAL, zorder=3, lw=0)
            ax.scatter([p], [yi], s=6, color=RED, zorder=4, lw=0)
            ax.text(121, yi, f"{p - c:+.1f}", va="center", ha="left", fontsize=4.6, clip_on=False,
                    color=RED if win else "0.45", fontweight="bold" if win else "normal")
        ax.set_xlim(0, 105)
        ax.set_xticks([0, 50, 100])
        ax.set_ylim(-0.6, len(ROWS) - 0.4)
        ax.set_yticks(yv, ROWS if j == 0 else [""] * len(ROWS))
        ax.tick_params(axis="y", length=0, labelsize=4.8)
        ax.set_title(ttl, fontsize=5.0, color=col, pad=1.5, fontweight="bold")
        ax.set_xlabel("accuracy (%)", fontsize=4.8, labelpad=0.5)
        ax.grid(axis="x", color="white", lw=0.5)
    C.bg.add_patch(Circle((x8 + 0.02, y2 + 0.03), 0.0045, fc=RED, ec="none"))
    C.text(x8 + 0.028, y2 + 0.03, "PhyQ-TransNet", fontsize=4.9, ha="left")
    C.bg.add_patch(Circle((x8 + 0.128, y2 + 0.03), 0.0045, fc=TEAL, ec="none"))
    C.text(x8 + 0.136, y2 + 0.03, "ClassBD", fontsize=4.9, ha="left")
    C.text(x8 + w8 - 0.008, y2 + 0.032, "SDUST & PU gains: p < 0.01", fontsize=4.9, ha="right", color=RED,
           fontweight="bold")

    fig.savefig(out / f"{a.name}.pdf")
    fig.savefig(out / f"{a.name}.png", dpi=600)
    from PIL import Image
    im = Image.open(out / f"{a.name}.png")
    im.convert("RGB").save(out / f"{a.name}.tif", compression="tiff_lzw", dpi=(600, 600))
    print("size", im.size, "peaks lin/sq white", round(S["linear_white_peak"], 1), round(S["square_white_peak"], 1),
          "sq imp", round(S["square_impulsive_peak"], 1))


if __name__ == "__main__":
    main()
