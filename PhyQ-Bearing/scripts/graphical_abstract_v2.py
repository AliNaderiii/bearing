#!/usr/bin/env python
"""Graphical abstract v2 for the MSSP submission (vector PDF + 600-dpi PNG/TIFF).

Every curve and number is data-derived:
  * waveform / spectra: phyq.sim.bearing_sim (identical generator, seed and filter as Fig. 2);
  * accuracies: user GPU results (review/user_results: main_tuned / main_benchmark), absolute SNR, test sets.
Native size 13.5 cm x 6 cm so that it is readable at the 13 cm width required by Elsevier (>= 1328 px).

    PHYQ_FONT_DIR=/path/to/fonts python scripts/graphical_abstract_v2.py --out <dir>
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

RED, TEAL, GREY, INK = "#D1495B", "#00798C", "#5D6D7E", "#1F2328"
BLUE, ORANGE = "#2C7FB8", "#E69F00"
Q_BG, A_BG, N_BG = "#FBE3E6", "#DDF0F3", "#F1F2F4"

# accuracy (%) at absolute SNR, mean over tasks/seeds (review/user_results/all_metrics.json, tab_main_values.json)
ROWS = ["SDUST", "PU 5-fold", "PU art→real", "IMS", "CWRU"]
GAUSS10 = {"SDUST": (66.28, 59.1), "PU 5-fold": (60.4, 43.9), "PU art→real": (47.0, 44.1),
           "IMS": (64.49, 71.74), "CWRU": (52.25, 64.72)}            # (PhyQ-TransNet, ClassBD)
IMP6 = {"SDUST": (42.7, 32.9), "PU 5-fold": (38.5, 35.1), "PU art→real": (31.7, 52.1),
        "IMS": (35.9, 41.0), "CWRU": (12.1, 40.5)}


def setup():
    d = os.environ.get("PHYQ_FONT_DIR")
    if d and os.path.isdir(d):
        for f in glob.glob(os.path.join(d, "**", "*.ttf"), recursive=True):
            font_manager.fontManager.addfont(f)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
        "font.size": 6.5, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2, "ytick.major.size": 2, "xtick.labelsize": 5.6, "ytick.labelsize": 5.6,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def sim_curves(n_avg=40, snr=0.0):
    """Exactly the computation of Fig. 2, Gaussian row (same rng sequence -> same numbers)."""
    from scipy.signal import firwin, lfilter
    from phyq.sim.bearing_sim import SimConfig, simulate_one
    fs = 12_000.0
    rng = np.random.default_rng(3)
    cfg = SimConfig()
    xs, ff = [], None
    for _ in range(n_avg):
        x, info = simulate_one("outer", rng, cfg, rpm=1797.0, return_info=True)
        xs.append(x)
        ff = info["f_fault"]
    X = np.stack(xs)
    n = rng.standard_normal(X.shape)
    p = (X ** 2).mean(-1, keepdims=True)
    N = n / np.sqrt((n ** 2).mean(-1, keepdims=True)) * np.sqrt(p / 10 ** (snr / 10))
    Xn = X + N
    h = firwin(15, [2_400, 4_200], pass_zero=False, fs=fs)
    Y = lfilter(h, 1.0, Xn, axis=-1)
    f = np.fft.rfftfreq(X.shape[-1], 1 / fs)
    sel = f <= 600
    out = {}
    for name, o in (("linear", Y), ("square", Y ** 2)):
        o = o - o.mean(-1, keepdims=True)
        P = (np.abs(np.fft.rfft(o * np.hanning(o.shape[-1]), axis=-1)) ** 2).mean(0)
        P = 10 * np.log10(P / np.median(P[sel & (f > 15)]))
        out[name] = P
        out[name + "_peak"] = P[np.abs(f - ff) < 4].max()
    return dict(fs=fs, f=f, ff=ff, clean=X[0], noisy=Xn[0], **out)


def box(bg, x, y, w, h, fc, ec, lw=0.6, r=0.012):
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                transform=bg.transAxes, fc=fc, ec=ec, lw=lw))


def arrow(bg, x0, x1, y, col="0.35"):
    bg.add_patch(FancyArrowPatch((x0, y), (x1, y), transform=bg.transAxes, arrowstyle="-|>",
                                 mutation_scale=7, lw=0.9, color=col))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "_paper_figures"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    setup()
    S = sim_curves()
    W, H = 13.5 / 2.54, 6.0 / 2.54
    fig = plt.figure(figsize=(W, H))
    bg = fig.add_axes([0, 0, 1, 1], zorder=-10)
    bg.axis("off")

    # ---------------- column frames and headers
    cols = [(0.008, 0.268, N_BG, GREY, "1  Problem"), (0.300, 0.318, Q_BG, RED, "2  Mechanism (theory)"),
            (0.650, 0.342, A_BG, TEAL, "3  Evidence: 5 datasets")]
    for x, w, fc, ec, t in cols:
        box(bg, x, 0.11, w, 0.775, fc, ec)
        fig.text(x + 0.012, 0.85, t, fontsize=7.0, fontweight="bold", color=ec, va="center")
    arrow(bg, 0.279, 0.297, 0.50)
    arrow(bg, 0.621, 0.647, 0.50)
    fig.text(0.5, 0.975, "PhyQ-TransNet: quadratic convolution as a learnable square-law envelope detector",
             ha="center", va="top", fontsize=8.0, fontweight="bold", color=INK)

    # ---------------- 1 problem: impacts buried in noise
    fs = S["fs"]
    nseg = int(0.06 * fs)
    t = np.arange(nseg) / fs * 1e3
    ax = fig.add_axes([0.03, 0.55, 0.23, 0.21])
    ax.plot(t, S["noisy"][:nseg], color="0.72", lw=0.35)
    ax.plot(t, S["clean"][:nseg], color=INK, lw=0.4)
    ax.set_yticks([])
    ax.set_xlabel("Time (ms)", fontsize=5.6, labelpad=1)
    ax.spines["left"].set_visible(False)
    ax.set_title("Fault impacts (black) + 0 dB noise (grey)", fontsize=5.8, pad=2, color=INK)
    fig.text(0.142, 0.36, "Fault information lives at the\nimpact repetition rate $f_o$ (Hz),\n"
             "but the energy sits in a kHz\nstructural resonance.", ha="center", va="center", fontsize=5.7,
             color=INK, linespacing=1.25)
    fig.text(0.142, 0.175, "A linear layer cannot move it to $f_o$\n(Proposition 1).", ha="center", va="center",
             fontsize=5.7, color=RED, fontweight="bold", linespacing=1.2)

    # ---------------- 2 mechanism: equation + spectra
    fig.text(0.459, 0.775, r"$y=(W_r\ast x)\odot(W_g\ast x)+W_b\ast(x\odot x)$", ha="center", va="center",
             fontsize=7.0, color=INK)
    fig.text(0.459, 0.715, "band-pass × band-pass → square-law detector (Prop. 2)", ha="center",
             va="center", fontsize=5.3, color=RED)
    f, ff = S["f"], S["ff"]
    sel = f <= 400
    for i, (key, lab, col) in enumerate((("linear", "Linear filter", GREY), ("square", "Quadratic (square law)", RED))):
        ax = fig.add_axes([0.357 + i * 0.14, 0.285, 0.112, 0.33])
        P = S[key]
        ax.fill_between(f[sel], -10, P[sel], color=col, alpha=0.18, lw=0)
        ax.plot(f[sel], P[sel], color=col, lw=0.6)
        for k in (1, 2, 3):
            ax.axvline(k * ff, color="0.35", lw=0.4, ls=(0, (2, 2)), zorder=0)
        ax.set_ylim(-10, 26)
        ax.set_xlim(0, 400)
        ax.set_xticks([0, 200, 400])
        ax.set_title(lab, fontsize=5.9, pad=2, color=col, fontweight="bold")
        ax.text(0.96, 0.95, f"$f_o$: {S[key + '_peak']:+.1f} dB", transform=ax.transAxes, ha="right", va="top",
                fontsize=5.6, color=col, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
        ax.set_xlabel("Frequency (Hz)", fontsize=5.4, labelpad=1)
        if i == 0:
            ax.set_ylabel("dB re. floor", fontsize=5.4, labelpad=1)
        else:
            ax.set_yticklabels([])
    fig.text(0.459, 0.14, "Cost: output variance grows with noise kurtosis (Prop. 3)", ha="center",
             va="center", fontsize=5.4, color=INK, style="italic")

    # ---------------- 3 evidence: dumbbells vs strongest baseline
    y = np.arange(len(ROWS))[::-1]
    for j, (data, ttl, col) in enumerate(((GAUSS10, "Gaussian −10 dB", BLUE), (IMP6, "Impulsive −6 dB", ORANGE))):
        ax = fig.add_axes([0.745 + j * 0.125, 0.265, 0.108, 0.47])
        for yi, r in zip(y, ROWS):
            p, c = data[r]
            win = p > c
            ax.plot([c, p], [yi, yi], color=RED if win else "0.6", lw=1.3, alpha=0.45, solid_capstyle="round")
            ax.scatter([c], [yi], s=9, color=TEAL, zorder=3, lw=0)
            ax.scatter([p], [yi], s=9, color=RED, zorder=4, lw=0)
            ax.text(113, yi, f"{p - c:+.1f}", va="center", ha="right", fontsize=5.2,
                    color=RED if win else "0.45", fontweight="bold" if win else "normal")
        ax.set_xlim(0, 115)
        ax.set_xticks([0, 50, 100])
        ax.set_ylim(-0.6, len(ROWS) - 0.4)
        ax.set_yticks(y, ROWS if j == 0 else [""] * len(ROWS))
        ax.tick_params(axis="y", length=0, labelsize=5.6)
        ax.set_title(ttl, fontsize=5.6, pad=2, color=col, fontweight="bold")
        ax.set_xlabel("Accuracy (%)", fontsize=5.4, labelpad=1)
        ax.grid(axis="x", color="white", lw=0.6)
        ax.set_facecolor("#F5FAFB")
    fig.text(0.668, 0.14, "●", color=RED, fontsize=6.5, va="center")
    fig.text(0.683, 0.14, "PhyQ-TransNet", fontsize=5.6, va="center", color=INK)
    fig.text(0.80, 0.14, "●", color=TEAL, fontsize=6.5, va="center")
    fig.text(0.815, 0.14, "ClassBD (best baseline)", fontsize=5.6, va="center", color=INK)

    # ---------------- bottom take-away
    fig.text(0.5, 0.055, "Gains where the noise is Gaussian-like (SDUST, PU, p < 0.01); losses under impulsive "
             "noise — predicted by the theory, reported openly.", ha="center", va="center", fontsize=6.0,
             color=INK)

    fig.savefig(out / "Graphical_abstract.pdf")
    fig.savefig(out / "Graphical_abstract.png", dpi=600)
    try:
        from PIL import Image
        im = Image.open(out / "Graphical_abstract.png")
        im.convert("RGB").save(out / "Graphical_abstract.tif", compression="tiff_lzw", dpi=(600, 600))
        print("size", im.size)
    except Exception as e:  # noqa: BLE001
        print("tiff skipped:", e)
    print("peaks", round(S["linear_peak"], 1), round(S["square_peak"], 1), "f_o", round(S["ff"], 1))


if __name__ == "__main__":
    main()
