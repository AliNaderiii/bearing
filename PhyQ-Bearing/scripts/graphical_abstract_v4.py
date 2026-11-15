#!/usr/bin/env python
"""Graphical abstract v4: four-stage story (PHYSICS -> THEORY -> MODEL -> EVIDENCE) + take-home ribbon.

Design goals (after reviewing recent MSSP graphical abstracts):
  * a physical object first (vector bearing drawing with an outer-race spall and accelerometer);
  * colour-coded stage columns with headers and chevrons (one reading direction);
  * every plot is data-derived: simulator spectra identical to Fig. 2, accuracies = user GPU results
    (absolute SNR, test sets, mean over tasks/seeds);
  * honest message: gain under Gaussian noise, failure mode under impulsive noise;
  * native 16 cm x 9 cm, vector PDF, 600-dpi PNG/TIFF (> 1328 px wide).

    PHYQ_FONT_DIR=/path/to/fonts python scripts/graphical_abstract_v4.py --out <dir> [--name NAME]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import graphical_abstract_v3 as v3  # noqa: E402  (reuses setup(), simulate() and the result tables)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import (Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Wedge, Circle as MCircle,  # noqa: E402
                                Rectangle, Arc)
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

W_CM, H_CM = 16.0, 9.0
ASPECT = W_CM / H_CM
NAVY, RED, TEAL, GREY, INK, AMBER, PURPLE = v3.NAVY, v3.RED, v3.TEAL, v3.GREY, v3.INK, v3.AMBER, v3.PURPLE
BLUE, ORANGE, GREEN = v3.BLUE, v3.ORANGE, "#2E9E5B"
STAGES = [  # (label, subtitle, strong colour, tint)
    ("PHYSICS", "what the sensor records", "#44546A", "#EEF2F6"),
    ("THEORY", "what a quadratic convolution computes", RED, "#FCEFF1"),
    ("MODEL", "PhyQ-TransNet", TEAL, "#E9F5F7"),
    ("EVIDENCE", "simulation + 5 real datasets", PURPLE, "#F3F0F9"),
]


def circ(xy, r, **kw):
    return Ellipse(xy, 2 * r, 2 * r * ASPECT, **kw)


class Canvas(v3.Canvas):
    def rbox(self, x, y, w, h, fc, ec, lw=0.5, r=0.008, z=1, alpha=1.0):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z,
                           alpha=alpha)
        self.bg.add_patch(p)
        return p

    def chevron(self, x, yc, col, w=0.016, h=0.075):
        pts = [(x, yc + h / 2), (x + w * 0.45, yc + h / 2), (x + w, yc), (x + w * 0.45, yc - h / 2), (x, yc - h / 2),
               (x + w * 0.55, yc)]
        self.bg.add_patch(Polygon(pts, closed=True, fc=col, ec="none", zorder=6, alpha=0.9))

    def chip(self, x, y, w, h, s, fc, ec, fs=4.9, bold=False, color=INK, z=3):
        self.rbox(x, y, w, h, fc, ec, lw=0.45, r=min(h / 2.2, 0.012), z=z)
        self.bg.text(x + w / 2, y + h / 2, s, fontsize=fs, ha="center", va="center", color=color, zorder=z + 1,
                     fontweight="bold" if bold else "normal", linespacing=1.0)

    def arr(self, p0, p1, col=GREY, lw=0.55, ms=5, conn="arc3", z=5):
        self.bg.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=col,
                                          connectionstyle=conn, zorder=z, shrinkA=0, shrinkB=0))


def gradient_ring(ax, c, r_out, r_in, c_out, c_in, n=14, z=1):
    """Annulus shaded radially (gives a turned-steel look)."""
    cmap = LinearSegmentedColormap.from_list("g", [c_in, c_out, c_in])
    rs = np.linspace(r_out, r_in, n + 1)
    for i in range(n):
        ax.add_patch(Wedge(c, rs[i], 0, 360, width=rs[i] - rs[i + 1], fc=cmap(i / (n - 1)), ec="none", zorder=z))


def ball(ax, c, r, z=4):
    cmap = LinearSegmentedColormap.from_list("b", ["#3E4A57", "#8C99A6", "#E9EEF2"])
    for k in range(12):
        t = k / 11
        rr = r * (1 - 0.82 * t)
        off = 0.38 * r * t
        ax.add_patch(MCircle((c[0] - off * 0.7, c[1] + off * 0.7), rr, fc=cmap(t), ec="none", zorder=z + k * 0.01))
    ax.add_patch(MCircle(c, r, fc="none", ec="#2F3A45", lw=0.35, zorder=z + 0.2))


SP = 52  # spall angle (deg)


def draw_bearing(ax):
    ax.set_aspect("equal")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.12, 1.45)
    ax.axis("off")
    c = (0.0, 0.0)
    # housing shadow
    ax.add_patch(MCircle((0.04, -0.05), 1.03, fc="#000000", alpha=0.10, ec="none", zorder=0))
    gradient_ring(ax, c, 1.0, 0.80, "#9AA6B2", "#DDE3E9", z=1)          # outer ring
    ax.add_patch(MCircle(c, 1.0, fc="none", ec="#4A5663", lw=0.5, zorder=2))
    ax.add_patch(MCircle(c, 0.80, fc="none", ec="#4A5663", lw=0.4, zorder=2))
    ax.add_patch(Wedge(c, 0.80, 0, 360, width=0.27, fc="#F7F9FB", ec="none", zorder=1))   # raceway gap
    ax.add_patch(Wedge(c, 0.672, 0, 360, width=0.026, fc="#C9A227", ec="none", alpha=0.85, zorder=3))  # cage
    gradient_ring(ax, c, 0.53, 0.36, "#9AA6B2", "#E3E8ED", z=1)          # inner ring
    ax.add_patch(MCircle(c, 0.53, fc="none", ec="#4A5663", lw=0.4, zorder=2))
    gradient_ring(ax, c, 0.36, 0.0, "#55606C", "#9AA4AE", n=10, z=1)     # shaft
    ax.add_patch(MCircle(c, 0.36, fc="none", ec="#2F3A45", lw=0.45, zorder=2))
    nb = 9
    for k in range(nb):
        a = np.pi / 2 + (k + 0.5) * 2 * np.pi / nb
        ball(ax, (0.665 * np.cos(a), 0.665 * np.sin(a)), 0.118)
    # rotation arrow on shaft
    ax.add_patch(Arc(c, 0.42, 0.42, theta1=200, theta2=480, color="white", lw=0.8, zorder=6))
    ax.annotate("", xy=(0.21 * np.cos(np.radians(125)), 0.21 * np.sin(np.radians(125))),
                xytext=(0.21 * np.cos(np.radians(140)), 0.21 * np.sin(np.radians(140))),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=0.8, mutation_scale=5), zorder=7)
    # outer-race spall at 12 o'clock (inner surface of outer ring)
    th = np.linspace(np.radians(SP - 9), np.radians(SP + 9), 9)
    rr = 0.80 + np.array([0.0, 0.05, 0.075, 0.05, 0.085, 0.06, 0.075, 0.045, 0.0])
    pts = [(0.80 * np.cos(t), 0.80 * np.sin(t)) for t in th] + [(r * np.cos(t), r * np.sin(t)) for t, r in
                                                                 zip(th[::-1], rr[::-1])]
    ax.add_patch(Polygon(pts, closed=True, fc=RED, ec="#7A1F2B", lw=0.3, zorder=8))
    # impact burst
    cs, sn = np.cos(np.radians(SP)), np.sin(np.radians(SP))
    for ang in np.radians([-50, -25, 0, 25, 50]):
        d = np.array([-cs * np.cos(ang) + sn * np.sin(ang), -sn * np.cos(ang) - cs * np.sin(ang)])
        p0 = np.array([0.76 * cs, 0.76 * sn])
        ax.plot([p0[0] + 0.05 * d[0], p0[0] + 0.15 * d[0]], [p0[1] + 0.05 * d[1], p0[1] + 0.15 * d[1]],
                color=AMBER, lw=0.7, zorder=9, solid_capstyle="round")
    # accelerometer on housing + vibration arcs
    ax.add_patch(FancyBboxPatch((-0.13, 1.0), 0.26, 0.2, boxstyle="round,pad=0,rounding_size=0.04", fc="#2B3A4A",
                                ec="#1A232D", lw=0.4, zorder=8))
    ax.add_patch(Rectangle((-0.05, 1.2), 0.10, 0.05, fc="#2B3A4A", ec="none", zorder=8))
    ax.plot([0.0, 0.10, 0.45, 0.75], [1.25, 1.36, 1.38, 1.30], color="#2B3A4A", lw=0.7, zorder=8,
            solid_capstyle="round")
    for k, rad in enumerate((0.22, 0.30, 0.38)):
        ax.add_patch(Arc((0.0, 1.1), rad * 2, rad * 2, theta1=20, theta2=70, color=TEAL, lw=0.55, alpha=1 - 0.25 * k,
                         zorder=8))
        ax.add_patch(Arc((0.0, 1.1), rad * 2, rad * 2, theta1=110, theta2=160, color=TEAL, lw=0.55, alpha=1 - 0.25 * k,
                         zorder=8))


def spectrum(ax, f, P, ff, col, ylim=(-10, 24), xt=(0, 200, 400), yt=(0, 20)):
    sel = f <= 400
    ax.fill_between(f[sel], ylim[0], P[sel], color=col, alpha=0.18, lw=0)
    ax.plot(f[sel], P[sel], color=col, lw=0.45)
    for kk in (1, 2, 3):
        ax.axvline(kk * ff, color="0.35", lw=0.3, ls=(0, (2, 2)), zorder=0)
    ax.set_ylim(*ylim)
    ax.set_xlim(0, 400)
    ax.set_xticks(list(xt))
    ax.set_yticks(list(yt))
    ax.tick_params(labelsize=4.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "results" / "_paper_figures"))
    ap.add_argument("--name", default="Graphical_abstract_v4")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    v3.setup()
    plt.rcParams["font.size"] = 5.2
    S = v3.simulate()
    f, ff, fs = S["f"], S["ff"], S["fs"]

    fig = plt.figure(figsize=(W_CM / 2.54, H_CM / 2.54))
    C = Canvas(fig)

    # ---------------------------------------------------------------- banner
    C.rbox(0.006, 0.9, 0.988, 0.092, NAVY, NAVY, r=0.012)
    C.rbox(0.006, 0.9, 0.988, 0.03, "#173C5E", "none", r=0.0, z=1.1, alpha=0.6)
    C.text(0.5, 0.958, "Quadratic convolution is a learnable square-law envelope detector",
           fontsize=8.2, color="white", fontweight="bold", zorder=3)
    C.text(0.5, 0.916, "PhyQ-TransNet for bearing fault diagnosis  •  theory  •  simulation with known fault frequency"
           "  •  5 leakage-controlled datasets", fontsize=5.3, color="#D6E2EE", zorder=3)

    # ---------------------------------------------------------------- stage columns
    y0, y1 = 0.14, 0.885
    cols_x = [(0.006, 0.188), (0.214, 0.262), (0.502, 0.166), (0.694, 0.300)]
    for i, ((x, w), (lab, sub, col, tint)) in enumerate(zip(cols_x, STAGES)):
        C.rbox(x, y0, w, y1 - y0, tint, col, lw=0.6, r=0.012, z=0.5)
        C.rbox(x, y1 - 0.062, w, 0.062, col, col, lw=0.6, r=0.012, z=0.6)
        C.rbox(x, y1 - 0.062, w, 0.03, col, "none", r=0.0, z=0.61)
        C.bg.add_patch(circ((x + 0.016, y1 - 0.031), 0.0105, fc="white", ec="none", zorder=1))
        C.text(x + 0.016, y1 - 0.031, str(i + 1), fontsize=6.0, color=col, fontweight="bold", zorder=2)
        C.text(x + 0.031, y1 - 0.022, lab, fontsize=6.4, color="white", fontweight="bold", ha="left", zorder=2)
        C.text(x + 0.031, y1 - 0.046, sub, fontsize=4.7, color="white", ha="left", zorder=2, alpha=0.95)
        if i < 3:
            nx = cols_x[i + 1][0]
            C.chevron(x + w + (nx - x - w) / 2 - 0.008, 0.53, col)

    # ---------------------------------------------------------------- 1 PHYSICS
    x, w = cols_x[0]
    axb = fig.add_axes([x + 0.008, 0.475, w - 0.016, 0.335])
    draw_bearing(axb)
    C.text(x + 0.012, 0.785, "accelerometer", fontsize=4.6, ha="left", color="#2B3A4A")
    C.chip(x + w - 0.058, 0.775, 0.054, 0.034, "outer-race\nspall", "white", RED, fs=4.4, color=RED, z=6)
    C.arr((x + w - 0.04, 0.775), (x + w * 0.66, 0.716), col=RED, lw=0.45, ms=4, conn="arc3,rad=-0.2")
    ax = fig.add_axes([x + 0.014, 0.275, w - 0.026, 0.14])
    n = int(0.05 * fs)
    t = np.arange(n) / fs * 1e3
    ax.plot(t, S["noisy_white"][:n], color="0.72", lw=0.3)
    ax.plot(t, S["clean"][:n], color=INK, lw=0.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 25, 50])
    ax.tick_params(labelsize=4.5)
    ax.set_xlabel("time (ms)", fontsize=4.6, labelpad=0.5)
    ax.patch.set_alpha(0)
    ax.annotate("", xy=(1e3 / ff * 2 + 1.0, ax.get_ylim()[1] * 0.92), xytext=(1e3 / ff + 1.0, ax.get_ylim()[1] * 0.92),
                arrowprops=dict(arrowstyle="<->", lw=0.45, color=RED, mutation_scale=4))
    C.text(x + 0.014 + (w - 0.026) * (1.5 / ff * 1e3 + 1) / 50, 0.43, "1/BPFO", fontsize=4.5, color=RED)
    C.text(x + w / 2, 0.19, "impacts at BPFO ring a kHz\nresonance (black), buried in\n0 dB noise (grey)",
           fontsize=4.8, linespacing=1.12)

    # ---------------------------------------------------------------- 2 THEORY
    x, w = cols_x[1]
    C.rbox(x + 0.01, 0.745, w - 0.02, 0.06, "white", RED, lw=0.5, r=0.008, z=2)
    C.text(x + w / 2, 0.786, "EQC layer", fontsize=4.6, color=RED, fontweight="bold", zorder=3)
    C.text(x + w / 2, 0.762, r"$y=(W_r\ast x)\odot(W_g\ast x)\;+\;W_b\ast(x\odot x)$", fontsize=6.0, zorder=3)
    # band-pass -> square -> lines
    ax = fig.add_axes([x + 0.016, 0.625, 0.07, 0.075])
    ax.fill_between(S["fir_f"] / 1e3, 0, S["fir_H"] / S["fir_H"].max(), color=RED, alpha=0.2, lw=0)
    ax.plot(S["fir_f"] / 1e3, S["fir_H"] / S["fir_H"].max(), color=RED, lw=0.6)
    ax.set_xlim(0, 6)
    ax.set_xticks([0, 3, 6])
    ax.set_yticks([])
    ax.tick_params(labelsize=4.4)
    ax.set_xticklabels(["0", "3", "6 kHz"])
    ax.patch.set_alpha(0)
    C.text(x + 0.051, 0.712, "band-pass", fontsize=4.6, color=RED, fontweight="bold")
    C.arr((x + 0.092, 0.66), (x + 0.108, 0.66), col=GREY)
    C.chip(x + 0.110, 0.635, 0.042, 0.05, r"$(\cdot)^2$", "#FFF3D6", AMBER, fs=7)
    C.text(x + 0.131, 0.712, "square law", fontsize=4.6, color=AMBER, fontweight="bold")
    C.arr((x + 0.155, 0.66), (x + 0.171, 0.66), col=GREY)
    C.chip(x + 0.173, 0.635, 0.079, 0.05, "squared\nenvelope", "white", RED, fs=4.7)
    C.text(x + 0.2125, 0.712, "one layer", fontsize=4.6, color=RED, fontweight="bold")
    # linear vs square spectra
    for k, (key, col, lab) in enumerate((("linear_white", GREY, "linear conv."), ("square_white", RED, "EQC square law"))):
        ax = fig.add_axes([x + 0.03 + k * 0.118, 0.415, 0.098, 0.1])
        spectrum(ax, f, S[key], ff, col, xt=(0, 200, 400), yt=(0, 20) if k == 0 else ())
        if k == 1:
            ax.spines["left"].set_visible(False)
        ax.set_xlabel("Hz", fontsize=4.5, labelpad=0.3)
        ax.set_title(f"{lab}\n{S[key + '_peak']:+.1f} dB", fontsize=4.7, color=col, pad=1.2, fontweight="bold", linespacing=1.0)
        ax.patch.set_alpha(0)
    C.text(x + w / 2, 0.578, "envelope spectra, 0 dB noise (dashed: BPFO harmonics)", fontsize=4.5, color=GREY,
           style="italic")
    props = [("P1", "linear stacks cannot create fault-frequency lines"),
             ("P2", "EQC product = band-pass + square-law detector"),
             ("P3", r"noise $\propto(\kappa_w-1)\,\sigma_w^4$  →  fragile to impulses")]
    for i, (pn, s) in enumerate(props):
        yy = 0.318 - i * 0.06
        C.rbox(x + 0.01, yy, w - 0.02, 0.048, "white", RED, lw=0.4, r=0.01, z=2)
        C.bg.add_patch(circ((x + 0.027, yy + 0.024), 0.0105, fc=RED, ec="none", zorder=3))
        C.text(x + 0.027, yy + 0.024, pn, fontsize=4.5, color="white", fontweight="bold", zorder=4)
        C.text(x + 0.043, yy + 0.024, s, fontsize=4.45, ha="left", zorder=4)

    # ---------------------------------------------------------------- 3 MODEL
    x, w = cols_x[2]
    cx = x + w / 2
    C.chip(x + 0.035, 0.765, w - 0.07, 0.036, "raw window  x", "white", GREY, fs=4.9, bold=True)
    lx, lw_, rx, rw = x + 0.01, 0.098, x + 0.114, w - 0.124
    C.arr((cx - 0.02, 0.765), (lx + lw_ / 2, 0.733), col=GREY)
    C.arr((cx + 0.02, 0.765), (rx + rw / 2, 0.733), col=GREY)
    deep = [("EQC stem", "#FBE3E6", RED), ("multi-scale\nEQC k31/15/7", "#FBE3E6", RED), ("QRes ×2\n(CA + soft thr.)", "#FBE3E6", RED),
            ("Transformer\n2 layers × 4 heads", "#DDF0F3", TEAL)]
    yy = 0.695
    for i, (s, fc, ec) in enumerate(deep):
        hh = 0.036 if "\n" not in s else 0.05
        C.chip(lx, yy - hh + 0.036, lw_, hh, s, fc, ec, fs=4.45)
        nxt = yy - hh + 0.036
        if i < len(deep) - 1:
            C.arr((lx + lw_ / 2, nxt), (lx + lw_ / 2, nxt - 0.012), col=GREY, ms=4)
        yy = nxt - 0.012 - 0.036
    ybot = nxt
    C.chip(rx, 0.53, rw, 0.198, "11\nsignal\nstatistics\nof x\n\nRMS,\nkurtosis,\ncrest, …", "#FDF0DC", AMBER, fs=4.3)
    C.arr((lx + lw_ / 2, ybot), (cx - 0.02, 0.402), col=GREY)
    C.arr((rx + rw / 2, 0.53), (cx + 0.03, 0.402), col=GREY)
    C.text(lx + lw_ / 2 - 0.004, 0.43, "$z_d$", fontsize=5.0, color=RED)
    C.text(rx + rw / 2 + 0.012, 0.47, "$z_s$", fontsize=5.0, color=AMBER)
    C.chip(x + 0.012, 0.36, w - 0.024, 0.042, r"gate  $(1-\alpha)\,z_d+\alpha\,z_s$", "#ECE6F5", PURPLE, fs=4.9)
    C.arr((cx, 0.36), (cx, 0.338), col=GREY)
    C.chip(x + 0.04, 0.302, w - 0.08, 0.036, "classifier", "white", GREY, fs=4.9, bold=True)
    for i, (s, c) in enumerate((("N", GREY), ("IR", RED), ("OR", AMBER), ("B", TEAL))):
        xx = x + 0.035 + i * (w - 0.07) / 3
        C.bg.add_patch(circ((xx, 0.262), 0.0115, fc=c, ec="none", zorder=3))
        C.text(xx, 0.262, s, fontsize=4.4, color="white", fontweight="bold", zorder=4)
    C.text(cx, 0.195, "1.40 M parameters\n8.6 ms / window (GTX 1650)", fontsize=4.6, color=GREY, linespacing=1.15)

    # ---------------------------------------------------------------- 4 EVIDENCE
    x, w = cols_x[3]
    C.rbox(x + 0.008, 0.6, w - 0.016, 0.21, "white", PURPLE, lw=0.4, r=0.01, z=1)
    C.text(x + 0.016, 0.79, "a  Simulation, square-law output: P3 confirmed", fontsize=4.9, color=PURPLE,
           fontweight="bold", ha="left")
    for k, (kind, col, lab) in enumerate((("white", BLUE, "Gaussian noise"), ("impulsive", ORANGE, "impulsive noise"))):
        ax = fig.add_axes([x + 0.03 + k * 0.142, 0.648, 0.112, 0.092])
        spectrum(ax, f, S[f"square_{kind}"], ff, col, yt=(0, 20) if k == 0 else ())
        if k == 1:
            ax.spines["left"].set_visible(False)
        ax.set_xlabel("Hz", fontsize=4.5, labelpad=0.3)
        ax.set_title(f"{lab}, 0 dB\npeak {S[f'square_{kind}_peak']:+.1f} dB", fontsize=4.6, color=col, pad=1.2,
                     fontweight="bold", linespacing=1.0)
        ax.patch.set_alpha(0)
    C.rbox(x + 0.008, 0.155, w - 0.016, 0.432, "white", PURPLE, lw=0.4, r=0.01, z=1)
    C.text(x + 0.016, 0.568, "b  Real data: PhyQ-TransNet vs ClassBD", fontsize=4.9, color=PURPLE,
           fontweight="bold", ha="left")
    C.text(x + 0.016, 0.535, "leakage-free splits • 3 seeds • 7 tuned baselines • Holm tests", fontsize=4.4,
           color=GREY, ha="left")
    yv = np.arange(len(v3.ROWS))[::-1]
    for j, (data, ttl, col) in enumerate(((v3.GAUSS10, "Gaussian −10 dB", BLUE), (v3.IMP6, "impulsive −6 dB", ORANGE))):
        ax = fig.add_axes([x + 0.072 + j * 0.112, 0.235, 0.076, 0.235])
        ax.set_facecolor("#F7F5FB")
        for yi, r in zip(yv, v3.ROWS):
            p, c = data[r]
            win = p > c
            ax.plot([c, p], [yi, yi], color=RED if win else "0.6", lw=1.2, alpha=0.45, solid_capstyle="round")
            ax.scatter([c], [yi], s=7, color=TEAL, zorder=3, lw=0)
            ax.scatter([p], [yi], s=7, color=RED, zorder=4, lw=0)
            ax.text(108, yi, f"{p - c:+.1f}", va="center", ha="left", fontsize=4.4, clip_on=False,
                    color=RED if win else "0.45", fontweight="bold" if win else "normal")
        ax.set_xlim(0, 105)
        ax.set_xticks([0, 50, 100])
        ax.set_ylim(-0.6, len(v3.ROWS) - 0.4)
        ax.set_yticks(yv, v3.ROWS if j == 0 else [""] * len(v3.ROWS))
        ax.tick_params(axis="y", length=0, labelsize=4.6)
        ax.tick_params(axis="x", labelsize=4.4)
        ax.set_title(ttl, fontsize=4.9, color=col, pad=1.5, fontweight="bold")
        ax.set_xlabel("accuracy (%)", fontsize=4.5, labelpad=0.4)
        ax.grid(axis="x", color="white", lw=0.6)
    C.bg.add_patch(circ((x + 0.02, 0.177), 0.0042, fc=RED, ec="none", zorder=3))
    C.text(x + 0.027, 0.177, "PhyQ-TransNet", fontsize=4.6, ha="left")
    C.bg.add_patch(circ((x + 0.112, 0.177), 0.0042, fc=TEAL, ec="none", zorder=3))
    C.text(x + 0.119, 0.177, "ClassBD", fontsize=4.6, ha="left")
    C.text(x + w - 0.012, 0.177, "SDUST, PU folds: p < 0.01", fontsize=4.3, ha="right", color=RED)

    # ---------------------------------------------------------------- take-home ribbon
    C.rbox(0.006, 0.012, 0.988, 0.108, "white", NAVY, lw=0.6, r=0.012)
    C.text(0.018, 0.066, "TAKE-\nHOME", fontsize=5.4, color=NAVY, fontweight="bold", ha="left", linespacing=1.0)
    C.rbox(0.075, 0.024, 0.44, 0.084, "#EAF4FB", BLUE, lw=0.45, r=0.01, z=2)
    C.bg.text(0.09, 0.066, "✔", fontsize=9, color=GREEN, ha="center", va="center", fontfamily="DejaVu Sans", zorder=3)
    C.text(0.105, 0.066, "Gaussian noise: PhyQ-TransNet is the most accurate model on SDUST\nand bearing-disjoint Paderborn at −10 dB (p < 0.01).", fontsize=5.0, ha="left", linespacing=1.15, zorder=3)
    C.rbox(0.528, 0.024, 0.456, 0.084, "#FDF1E2", ORANGE, lw=0.45, r=0.01, z=2)
    C.bg.text(0.543, 0.066, "✘", fontsize=9, color=RED, ha="center", va="center", fontfamily="DejaVu Sans", zorder=3)
    C.text(0.558, 0.066, "Impulsive noise: fragile, exactly as P3 predicts. Saturating the\nsquare law (as in ClassBD) "
           "restores robustness in simulation.", fontsize=5.0, ha="left", linespacing=1.15, zorder=3)

    fig.savefig(out / f"{a.name}.pdf")
    fig.savefig(out / f"{a.name}.png", dpi=600)
    from PIL import Image
    im = Image.open(out / f"{a.name}.png")
    im.convert("RGB").save(out / f"{a.name}.tif", compression="tiff_lzw", dpi=(600, 600))
    print("size", im.size)


if __name__ == "__main__":
    main()
