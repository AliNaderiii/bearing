#!/usr/bin/env python
"""Figure 1 of the MSSP manuscript: architecture of PhyQ-TransNet (final configuration, no Neural MED).

    python scripts/arch_figure.py                 # -> results/_paper_figures/Figure_1_architecture.pdf/.png

Drawn with matplotlib (vector PDF). Every block, size and shape corresponds to phyq/models/phyq_paper.py
(PhyQTransNet with prefilter.enabled = false, input_norm = instance); tensor shapes are for a window of
L samples and C channels (L = 2048 for CWRU/IMS/NREL/SDUST, 8192 for PU).
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# palette shared with scripts/pro_figures.py
Q_FC, Q_EC = "#FBE3E6", "#D1495B"        # quadratic (EQC) blocks
T_FC, T_EC = "#DDF0F3", "#00798C"        # attention / Transformer
S_FC, S_EC = "#FDF0DC", "#D98E1F"        # statistical stream
N_FC, N_EC = "#F1F2F4", "#5D6D7E"        # neutral (norm, pooling, linear)
F_FC, F_EC = "#ECE6F5", "#6C4FA3"        # fusion / head
TXT = "#1B1B1B"
SHAPE = "#6B6B6B"


def setup():
    extra = os.environ.get("PHYQ_FONT_DIR")
    if extra and os.path.isdir(extra):
        for f in glob.glob(os.path.join(extra, "**", "*.ttf"), recursive=True):
            font_manager.fontManager.addfont(f)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic",
        "mathtext.bf": "sans:bold", "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


class Canvas:
    def __init__(self, ax):
        self.ax = ax

    def box(self, x, y, w, h, text, fc=N_FC, ec=N_EC, fs=6.2, bold=False, lw=0.8, ls="-", color=TXT,
            rs=0.9, z=2, va="center", ha="center", tx=None, ty=None):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rs}", fc=fc, ec=ec, lw=lw,
                           ls=ls, zorder=z)
        self.ax.add_patch(p)
        if text:
            self.ax.text(x + w / 2 if tx is None else tx, y + h / 2 if ty is None else ty, text, ha=ha, va=va,
                         fontsize=fs, color=color, fontweight="bold" if bold else "normal", zorder=z + 1,
                         linespacing=1.18)
        return (x, y, w, h)

    def arrow(self, p0, p1, color="#4A4A4A", lw=0.8, cs="arc3", ms=6, z=1, ls="-"):
        a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                            connectionstyle=cs, zorder=z, shrinkA=0, shrinkB=0, linestyle=ls)
        self.ax.add_patch(a)

    def line(self, xs, ys, color="#4A4A4A", lw=0.8, z=1, ls="-"):
        self.ax.plot(xs, ys, color=color, lw=lw, zorder=z, solid_capstyle="butt", ls=ls)

    def text(self, x, y, s, fs=5.6, color=SHAPE, ha="center", va="center", **kw):
        self.ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, zorder=5, **kw)

    def op(self, x, y, sym, r=1.35, fc="white", ec="#4A4A4A", fs=7.5):
        self.ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=0.8, zorder=3))
        self.ax.text(x, y - 0.05, sym, ha="center", va="center", fontsize=fs, zorder=4, color=TXT)


def right(b):
    return (b[0] + b[2], b[1] + b[3] / 2)


def left(b):
    return (b[0], b[1] + b[3] / 2)


def top(b):
    return (b[0] + b[2] / 2, b[1] + b[3])


def bottom(b):
    return (b[0] + b[2] / 2, b[1])


def draw(out: Path):
    setup()
    W, H = 100.0, 64.0
    fig = plt.figure(figsize=(7.0, 7.0 * H / W))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    c = Canvas(ax)

    # ================================================================ (a) overall architecture
    c.text(0.6, 62.6, "(a)", fs=8.5, color=TXT, ha="left", fontweight="bold")
    c.text(4.2, 62.6, "PhyQ-TransNet (final configuration)", fs=7.2, color=TXT, ha="left", fontweight="bold")
    c.text(39.5, 62.55, "the Neural-MED pre-filter of the original design is disabled (validation ablation)", fs=5.4,
           ha="left")
    # stream labels
    c.text(13.4, 58.0, "DEEP STREAM", fs=5.8, color=Q_EC, ha="left", fontweight="bold")
    c.text(13.4, 31.5, "STATISTICAL STREAM", fs=5.8, color=S_EC, ha="left", fontweight="bold")
    # background bands
    c.box(12.4, 37.6, 74.6, 22.4, "", fc="#FFF8F9", ec="#F3C7CE", lw=0.5, z=0, rs=1.5)
    c.box(12.4, 23.2, 74.6, 9.4, "", fc="#FFFBF4", ec="#F2DDB8", lw=0.5, z=0, rs=1.5)

    yD, hD = 44.0, 11.0
    # input
    b_in = c.box(0.8, 33.0, 9.6, 16.0, "Noisy\nvibration\nwindow\n\n$x\\in\\mathbb{R}^{C\\times L}$",
                 fc="white", ec="#4A4A4A", fs=6.2)
    # tiny waveform icon
    import numpy as np
    t = np.linspace(0, 1, 160)
    sig = 0.55 * np.sin(2 * np.pi * 38 * t) * (np.exp(-((t * 5) % 1) * 6)) + 0.12 * np.random.default_rng(1).standard_normal(160)
    ax.plot(1.8 + 7.6 * t, 35.6 + 1.3 * sig, color=Q_EC, lw=0.5, zorder=4)

    xs = 13.4
    gap = 2.1
    b_norm = c.box(xs, yD, 7.4, hD, "Instance\nnorm.\n(per window,\nper channel)", fs=5.6)
    c.text(xs + 3.7, yD - 1.6, "C×L")
    x = xs + 7.4 + gap
    b_stem = c.box(x, yD, 8.6, hD, "EQC stem\nk 15, s 2\n32 ch\nBN · ACON\nmax-pool", fc=Q_FC, ec=Q_EC, fs=5.6)
    c.text(x + 4.3, yD - 1.6, "32×L/4")
    x += 8.6 + gap
    # multi-scale block
    b_ms = c.box(x, yD, 13.2, hD, "", fc="#FFF1F3", ec=Q_EC, lw=0.8)
    c.text(x + 6.6, yD + hD - 1.3, "Multi-scale EQC", fs=5.7, color=TXT, fontweight="bold")
    for i, k in enumerate((31, 15, 7)):
        c.box(x + 0.8 + i * 3.25, yD + 2.6, 2.9, 5.3, f"k {k}\n64", fc=Q_FC, ec=Q_EC, fs=5.2, rs=0.5, z=3)
    c.text(x + 11.25, yD + 5.25, "avg", fs=5.4, color=TXT)
    c.text(x + 6.6, yD + 1.3, "BN · ACON · max-pool", fs=5.0, color=SHAPE)
    c.text(x + 6.6, yD - 1.6, "64×L/8")
    x += 13.2 + gap
    b_pw = c.box(x, yD, 5.0, hD, "1×1\nconv\n→128", fs=5.6)
    c.text(x + 2.5, yD - 1.6, "128×L/8")
    x += 5.0 + gap
    b_res = c.box(x, yD, 9.6, hD, "2 × QResBlock-\nPro\n(EQC, coord.\nattention, soft\nthreshold.)", fc=Q_FC,
                  ec=Q_EC, fs=5.5)
    c.text(x + 4.8, yD - 1.6, "128×L/16")
    x += 9.6 + gap
    b_tf = c.box(x, yD, 9.6, hD, "Transformer\nencoder\n2 layers · 4 heads\nd 128 · FF 512\n+ pos. emb.",
                 fc=T_FC, ec=T_EC, fs=5.5)
    c.text(x + 4.8, yD - 1.6, "L/16 tokens")
    x += 9.6 + gap
    b_gap = c.box(x, yD + 2.0, 5.2, 7.0, "mean\nover\ntokens", fs=5.4)
    c.text(x + 2.6, yD - 1.6, "$z_d\\in\\mathbb{R}^{128}$", fs=6.0, color=Q_EC)

    for a, b in ((b_norm, b_stem), (b_stem, b_ms), (b_ms, b_pw), (b_pw, b_res), (b_res, b_tf)):
        c.arrow(right(a), left(b))
    c.arrow(right(b_tf), left(b_gap))
    # input -> norm
    c.line([b_in[0] + b_in[2], 12.0], [yD + hD / 2, yD + hD / 2])
    c.arrow((12.0, yD + hD / 2), left(b_norm))

    # statistical stream
    yS, hS = 23.9, 6.6
    b_st = c.box(13.4, yS, 26.4, hS, "11 statistics per channel of the raw noisy input:\nmean, s.d., RMS, skewness, kurtosis, peak, crest and\n"
                 "shape factors, spectral energy, envelope mean and s.d.",
                 fc=S_FC, ec=S_EC, fs=5.2)
    b_sc = c.box(41.9, yS, 10.0, hS, "Standardise\n(training\nstatistics)", fc=S_FC, ec=S_EC, fs=5.5)
    b_mlp = c.box(54.0, yS, 15.6, hS, "MLP  11C → 64 → 128\nBN · SiLU · dropout 0.3", fc=S_FC, ec=S_EC, fs=5.5)
    c.text(78.4, yS + hS / 2 + 1.3, "$z_s\\in\\mathbb{R}^{128}$", fs=6.0, color=S_EC)
    c.arrow(right(b_st), left(b_sc))
    c.arrow(right(b_sc), left(b_mlp))
    c.line([b_in[0] + b_in[2], 12.0], [yS + hS / 2, yS + hS / 2])
    c.line([5.6, 5.6], [b_in[1], yS + hS / 2 + 0.0], lw=0)  # (spacer)
    c.arrow((12.0, yS + hS / 2), left(b_st))

    # fusion head
    fx, fw = 88.0, 11.6
    b_fu = c.box(fx, 26.0, fw, 31.0, "", fc=F_FC, ec=F_EC, lw=0.9, rs=1.2)
    c.text(fx + fw / 2, 55.2, "Gated fusion", fs=6.0, color=F_EC, fontweight="bold")
    c.text(fx + fw / 2, 51.4, "gate (weight of the\nstatistical stream)", fs=5.0, color=SHAPE)
    c.text(fx + fw / 2, 47.6, "$\\alpha=\\sigma(W[z_d;z_s]+b)$", fs=5.4, color=TXT)
    c.text(fx + fw / 2, 43.0, "$z=(1-\\alpha)\\,z_d+\\alpha\\,z_s$", fs=5.4, color=TXT)
    b_head = c.box(fx + 0.9, 31.6, fw - 1.8, 7.4, "LayerNorm\ndropout 0.3\nlinear", fc="white", ec=F_EC, fs=5.4)
    c.arrow(bottom((fx, 39.4, fw, 1.0)), top(b_head), color=F_EC)
    b_out = c.box(fx + 0.9, 27.0, fw - 1.8, 3.4, "K classes", fc=F_EC, ec=F_EC, fs=5.8, color="white", bold=True)
    c.arrow(bottom(b_head), top(b_out), color=F_EC)
    # z_d / z_s into fusion
    c.arrow(right(b_gap), (fx, yD + hD / 2), color=Q_EC, lw=1.0)
    c.arrow(right(b_mlp), (fx, yS + hS / 2), color=S_EC, lw=1.0)

    # ================================================================ (b) EQC
    c.text(0.6, 19.8, "(b)", fs=8.5, color=TXT, ha="left", fontweight="bold")
    c.text(4.2, 19.8, "Enhanced quadratic convolution (EQC)", fs=7.2, color=TXT, ha="left", fontweight="bold")
    c.box(0.6, 0.6, 46.0, 17.6, "", fc="#FFF8F9", ec="#F3C7CE", lw=0.5, z=0, rs=1.5)
    bx = c.box(1.8, 7.0, 4.4, 5.0, "$x$", fc="white", ec="#4A4A4A", fs=7)
    b_r = c.box(10.8, 13.0, 8.2, 3.8, "$W_r * x + b_r$", fc=Q_FC, ec=Q_EC, fs=5.8)
    b_g = c.box(10.8, 7.6, 8.2, 3.8, "$W_g * x + b_g$", fc=Q_FC, ec=Q_EC, fs=5.8)
    c.op(9.0, 4.0, "$x^2$", r=1.4, fs=5.4)
    b_b = c.box(10.8, 2.1, 8.2, 3.8, "$W_b * (x\\odot x)$", fc=Q_FC, ec=Q_EC, fs=5.8)
    # fan-out
    c.line([6.2, 7.6], [9.5, 9.5])
    c.line([7.6, 7.6], [4.0, 14.9])
    c.arrow((7.6, 14.9), left(b_r))
    c.arrow((7.6, 9.5), left(b_g))
    c.arrow((7.6, 4.0), (9.0 - 1.4, 4.0))
    c.arrow((9.0 + 1.4, 4.0), left(b_b))
    c.op(23.0, 12.2, "$\\odot$", fs=8)
    c.arrow(right(b_r), (23.0 - 1.0, 12.2 + 0.9), cs="arc3,rad=-0.15")
    c.arrow(right(b_g), (23.0 - 1.0, 12.2 - 0.9), cs="arc3,rad=0.15")
    c.op(28.0, 8.0, "$+$", fs=8)
    c.arrow((23.0 + 0.95, 12.2 - 0.95), (28.0 - 0.95, 8.0 + 0.95))
    c.arrow(right(b_b), (28.0 - 1.0, 8.0 - 0.95), cs="arc3,rad=0.2")
    c.text(28.0, 11.2, "+ c", fs=5.6, color=TXT)
    b_y = c.box(31.4, 6.0, 3.4, 4.0, "$y$", fc="white", ec="#4A4A4A", fs=7)
    c.arrow((28.0 + 1.35, 8.0), left(b_y))
    c.text(41.0, 14.2, "bilinear term =\nband-pass filter\n+ square law\n(Proposition 2)", fs=5.5, color=Q_EC,
           fontweight="bold")
    c.text(35.0, 2.3, "$y=(W_r*x+b_r)\\odot(W_g*x+b_g)+W_b*(x\\odot x)+c$", fs=5.4, color=TXT)
    c.arrow((37.4, 12.6), (24.4, 12.8), color=Q_EC, lw=0.6, cs="arc3,rad=0.25", ls="--")

    # ================================================================ (c) QResBlock-Pro
    c.text(48.6, 19.8, "(c)", fs=8.5, color=TXT, ha="left", fontweight="bold")
    c.text(52.2, 19.8, "QResBlock-Pro (residual block)", fs=7.2, color=TXT, ha="left", fontweight="bold")
    c.box(48.6, 0.6, 50.8, 17.6, "", fc="#FFF8F9", ec="#F3C7CE", lw=0.5, z=0, rs=1.5)
    items = [("EQC\nk 3", Q_FC, Q_EC), ("BN\nACON", N_FC, N_EC), ("EQC\nk 3", Q_FC, Q_EC), ("BN", N_FC, N_EC),
             ("Coord.\natten-\ntion", T_FC, T_EC), ("Soft\nthres-\nhold", T_FC, T_EC)]
    x0, y0, bw, bh, g2 = 52.5, 5.6, 4.9, 7.4, 1.3
    prev = None
    c.box(49.4, y0 + 1.7, 2.4, 4.0, "in", fc="white", ec="#4A4A4A", fs=5.4)
    for i, (s, fc, ec) in enumerate(items):
        b = c.box(x0 + i * (bw + g2), y0, bw, bh, s, fc=fc, ec=ec, fs=5.4)
        if prev:
            c.arrow(right(prev), left(b))
        else:
            c.arrow((51.8, y0 + bh / 2), left(b))
        prev = b
    ox = x0 + len(items) * (bw + g2) + 0.9
    c.op(ox, y0 + bh / 2, "$+$", fs=8)
    c.arrow(right(prev), (ox - 1.35, y0 + bh / 2))
    b_a2 = c.box(ox + 2.4, y0 + 1.4, 4.4, 4.6, "ACON", fc=N_FC, ec=N_EC, fs=5.4)
    c.arrow((ox + 1.35, y0 + bh / 2), left(b_a2))
    # skip connection
    c.line([50.6, 50.6], [y0 + 1.7 + 4.0, 16.0])
    c.line([50.6, ox], [16.0, 16.0])
    c.arrow((ox, 16.0), (ox, y0 + bh / 2 + 1.35))
    c.text((50.6 + ox) / 2, 16.9, "identity skip connection", fs=5.2)
    c.text(74.0, 2.6,
           "soft thresholding: $\\mathrm{sign}(x)\\,\\max(0,|x|-\\tau_c)$, $\\tau_c$ predicted per channel", fs=5.0)

    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "Figure_1_architecture.pdf")
    fig.savefig(out / "Figure_1_architecture.png", dpi=600)
    plt.close(fig)
    print("  wrote", out / "Figure_1_architecture.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "_paper_figures"))
    draw(Path(ap.parse_args().out))
