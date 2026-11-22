#!/usr/bin/env python
"""Publication-quality figures for the MSSP manuscript (matplotlib + seaborn, vector PDF + 600-dpi PNG).

    python scripts/pro_figures.py                       # all figures
    python scripts/pro_figures.py --only snr heatmap    # a subset
    python scripts/pro_figures.py --results D:/bearing_repo/PhyQ-Bearing/results

Reads only result files inside the project (results/main_tuned, results/main_benchmark, results/pu_protocols,
results/sim_mechanism) and writes ONLY to  results/_paper_figures/  (nothing else is touched).
The figures that need test_probs.npz (forest plot of IMS/SDUST) fall back to McNemar-free seed-level
intervals when the npz files are not present.

Figures
  fig2_detectors    simulated spectra after linear / ReLU / square-law detection (Gaussian vs impulsive)
  fig3_prominence   fault-frequency prominence inside trained networks (dumbbell plot)
  fig4_kurtosis     Proposition 3: square-law output variance vs noise kurtosis + simulated accuracies
  fig5_snr          accuracy vs SNR, six dataset groups, mean +/- s.d. bands
  fig6_heatmap      accuracy under white / pink / impulsive noise (-6 dB), all models x all datasets
  fig7_forest       paired accuracy differences of PhyQ-TransNet vs every baseline (-6 and -10 dB)
  fig8_confusion    confusion matrices, SDUST and PU art2real at -10 dB, PhyQ-TransNet vs ClassBD
  fig9_leakage      PU: seen bearings vs unseen bearings vs artificial-to-real (protocol effect)
  fig10_interp      REAL data: learned stem filters, feature-map baseband spectra, fault-line prominence
  fig11_interp      REAL data: fusion gate alpha, activation growth through depth, t-SNE of the fused z
                    (both need results/interp/_analysis/*.npz from scripts/interpret.py analyze)
  interactive       Interactive_results_explorer.html (plotly, self-contained) for the supplementary material
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import seaborn as sns  # noqa: E402

# ----------------------------------------------------------------------------------------------- style
TEXTWIDTH = 7.0  # inches (Elsevier full width 190 mm, cas-sc text block ~ 6.9 in)


def setup_style():
    extra = os.environ.get("PHYQ_FONT_DIR")
    if extra and os.path.isdir(extra):
        for f in glob.glob(os.path.join(extra, "**", "*.ttf"), recursive=True):
            font_manager.fontManager.addfont(f)
    sns.set_theme(context="paper", style="ticks")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic",
        "mathtext.bf": "sans:bold",
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8, "legend.fontsize": 6.8, "legend.title_fontsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "lines.linewidth": 1.0, "lines.markersize": 3.2, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.5,
        "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150,
    })


# one colour / marker / line style per model, used in EVERY figure (Okabe-Ito based, colour-blind safe)
MODELS = [  # tag, label, colour, marker, linestyle
    ("phyq_nomed", "PhyQ-TransNet", "#D1495B", "o", "-"),
    ("phyq_nomed_noquad", "Linear twin", "#EDA03B", "s", (0, (4, 1.5))),
    ("phyq_nomed_noquad_wide", "Linear twin, width-matched", "#8B4513", "d", (0, (5, 1.2, 1, 1.2))),  # v10.10f
    ("classbd", "ClassBD", "#00798C", "D", "-"),
    ("cnn_transformer", "CNN–Transformer", "#56B4E9", "^", "-"),
    ("drsn", "DRSN-CW", "#2E9E5B", "v", "-"),
    ("qcnn", "QCNN", "#9B59B6", "P", "-"),
    ("wdcnn", "WDCNN", "#5D6D7E", "X", "-"),
    ("stats_mlp", "Statistics MLP", "#A9A9A9", "h", (0, (1, 1))),
    ("kurtogram_ses", "Fast Kurtogram + SES", "#1B1B1B", "*", (0, (3, 1, 1, 1))),
]
LABEL = {m[0]: m[1] for m in MODELS}
TWINS = ["phyq_nomed_noquad", "phyq_nomed_noquad_wide"]  # linear controls of the proposed model
COLOR = {m[0]: m[2] for m in MODELS}
MARK = {m[0]: m[3] for m in MODELS}
LS = {m[0]: m[4] for m in MODELS}
LABEL["phyq_transnet"] = "PhyQ-TransNet + MED (original)"
COLOR["phyq_transnet"] = "#8E2C3A"
MARK["phyq_transnet"] = "o"
LS["phyq_transnet"] = "-"
PROPOSED = "phyq_nomed"
GROUPS = ["CWRU", "IMS", "NREL", "SDUST", "PU 5-fold", "PU art2real"]
CHANCE = {"CWRU": 10, "IMS": 25, "NREL": 50, "SDUST": 10, "PU 5-fold": 100 / 3, "PU art2real": 100 / 3}
NCLS = {"CWRU": 10, "IMS": 4, "NREL": 2, "SDUST": 10, "PU 5-fold": 3, "PU art2real": 3}


def panel_label(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom", ha="right")


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=600)
    plt.close(fig)
    print("  wrote", out / f"{name}.pdf")


# ----------------------------------------------------------------------------------------------- data
def group_of(dataset, variant):
    if dataset == "cwru" and re.fullmatch(r"within_\dHP", variant):
        return "CWRU"
    if dataset == "pu" and re.fullmatch(r"fold\d", variant):
        return "PU 5-fold"
    if dataset == "pu" and variant == "art2real":
        return "PU art2real"
    if dataset in ("ims", "nrel", "sdust"):
        return dataset.upper()
    return None


def load_runs(results: Path, exps=("main_tuned", "main_benchmark")):
    """long table: one row per (exp, group, variant, model, seed, noise, mode, snr) with acc."""
    rows, conf = [], {}
    for rank, exp in enumerate(exps):
        for p in glob.glob(str(results / exp / "*" / "*" / "*" / "seed*" / "metrics.json")):
            parts = Path(p).parts
            ds, var, tag, seed = parts[-5], parts[-4], parts[-3], int(parts[-2][4:])
            g = group_of(ds, var)
            if g is None:
                continue
            d = json.loads(Path(p).read_text())
            if "test" not in d:
                continue
            t = d["test"]
            base = dict(rank=rank, exp=exp, group=g, variant=var, model=tag, seed=seed)
            rows.append({**base, "noise": "clean", "mode": "", "snr": np.nan, "acc": t["clean"]["acc"]})
            conf[(exp, g, var, tag, seed, "clean", "", None)] = t["clean"].get("confusion")
            for nz, modes in t.get("noise", {}).items():
                for mode, snrs in modes.items():
                    for s, v in snrs.items():
                        rows.append({**base, "noise": nz, "mode": mode, "snr": float(s), "acc": v["acc"]})
                        conf[(exp, g, var, tag, seed, nz, mode, float(s))] = v.get("confusion")
    df = pd.DataFrame(rows)
    # never pool experiments: per (group, model) keep the first experiment in `exps` that has it
    first = df.groupby(["group", "model"])["rank"].transform("min")
    df = df[df["rank"] == first].drop(columns="rank")
    return df, conf


def cond(df, noise, mode="absolute", snr=None):
    if noise == "clean":
        return df[df.noise == "clean"]
    return df[(df.noise == noise) & (df["mode"] == mode) & (df.snr == float(snr))]


def group_stats(d):
    """mean over tasks per seed, then mean / s.d. over seeds."""
    ps = d.groupby(["group", "model", "seed"]).acc.mean().reset_index()
    return ps.groupby(["group", "model"]).acc.agg(["mean", "std", "count"]).reset_index()


# ----------------------------------------------------------------------------------------------- fig 2
def fig2_detectors(out, n_avg=40, snr=0.0):
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

    def noise(x, kind):
        if kind == "white":
            n = rng.standard_normal(x.shape)
        else:
            n = 0.3 * rng.standard_normal(x.shape) + (rng.random(x.shape) < 0.01) * rng.standard_normal(x.shape) * 10
        p = (x ** 2).mean(-1, keepdims=True)
        return n / np.sqrt((n ** 2).mean(-1, keepdims=True)) * np.sqrt(p / 10 ** (snr / 10))

    h = firwin(15, [2_400, 4_200], pass_zero=False, fs=fs)
    dets = [("Linear filter only", lambda y: y), ("ReLU (linear law)", lambda y: np.maximum(y, 0)),
            ("Square law (quadratic)", lambda y: y ** 2)]
    f = np.fft.rfftfreq(X.shape[-1], 1 / fs)
    sel = f <= 600
    rowcol = {"white": "#2C7FB8", "impulsive": "#D1495B"}
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, 3.5), sharex=True, sharey=True)
    letters = iter("abcdef")
    for r, kind in enumerate(("white", "impulsive")):
        Xn = X + noise(X, kind)
        Y = lfilter(h, 1.0, Xn, axis=-1)
        for c, (name, fn) in enumerate(dets):
            o = fn(Y)
            o = o - o.mean(-1, keepdims=True)
            P = (np.abs(np.fft.rfft(o * np.hanning(o.shape[-1]), axis=-1)) ** 2).mean(0)
            floor = np.median(P[sel & (f > 15)])
            P = 10 * np.log10(P / floor)
            ax = axes[r, c]
            ax.grid(False)
            ax.fill_between(f[sel], -10, P[sel], color=rowcol[kind], alpha=0.18, lw=0)
            ax.plot(f[sel], P[sel], color=rowcol[kind], lw=0.8)
            proms = []
            for k in (1, 2, 3):
                m = np.abs(f - k * ff) < 4
                proms.append(P[m].max())
                ax.axvline(k * ff, color="0.35", lw=0.5, ls=(0, (2, 2)), zorder=0)
                if r == 0 and c == 0:
                    ax.text(k * ff, 26.5, ["$f_o$", "$2f_o$", "$3f_o$"][k - 1], ha="center", va="bottom",
                            fontsize=6.5, color="0.25")
            ax.text(0.97, 0.93, f"peak at $f_o$: {proms[0]:.1f} dB", transform=ax.transAxes, ha="right",
                    va="top", fontsize=6.3,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", lw=0.5))
            ax.set_ylim(-10, 26)
            ax.set_xlim(0, 600)
            if r == 0:
                ax.set_title(name, pad=12 if c == 0 else 4)
            if c == 0:
                ax.set_ylabel(("Gaussian" if kind == "white" else "Impulsive") + " noise, 0 dB\nPSD (dB re floor)")
            if r == 1:
                ax.set_xlabel("Frequency (Hz)")
            panel_label(ax, next(letters))
    sns.despine(fig)
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    save(fig, out, "Figure_2_detectors")


# ----------------------------------------------------------------------------------------------- fig 3
def fig3_prominence(results, out):
    p = results / "sim_mechanism" / "prominence.json"
    if not p.exists():
        print("  skip fig3 (no prominence.json)")
        return
    rows = json.loads(p.read_text())
    agg = defaultdict(list)
    for r in rows:
        v = r["top10_db"] if r["layer"] != "input" else r["mean_db"]
        agg[(r["snr"], r["model"], r["layer"])].append(v)
    items = [("Input", "Raw input signal", ("raw_input", "input")),
             ("Linear twin", "pre-activation", ("noquad", "pre")),
             ("Linear twin", "post-activation", ("noquad", "post")),
             ("Quadratic", "product term $(W_r x)\\odot(W_g x)$", ("quad", "product_rg")),
             ("Quadratic", "power term $W_b(x\\odot x)$", ("quad", "power_b")),
             ("Quadratic", "layer output, pre-BN", ("quad", "pre")),
             ("Quadratic", "post-activation", ("quad", "post")),
             ("Quadratic", "random initialisation", ("quad_init", "pre"))]
    fam_col = {"Input": "#5D6D7E", "Linear twin": "#EDA03B", "Quadratic": "#D1495B"}
    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.62, 2.7))
    y = np.arange(len(items))[::-1].astype(float)
    for yi, (fam, name, (m, l)) in zip(y, items):
        c = [v for v in agg.get(("clean", m, l), []) if np.isfinite(v)]
        n = [v for v in agg.get((0, m, l), []) if np.isfinite(v)]
        mc, mn = np.mean(c), np.mean(n)
        ax.plot([mn, mc], [yi, yi], color=fam_col[fam], lw=2.2, alpha=0.35, solid_capstyle="round", zorder=1)
        ax.scatter(n, np.full(len(n), yi), s=5, color=fam_col[fam], alpha=0.35, lw=0, zorder=2)
        ax.scatter(c, np.full(len(c), yi), s=5, color="k", alpha=0.25, lw=0, zorder=2)
        ax.scatter([mc], [yi], s=34, color=fam_col[fam], edgecolor="k", lw=0.5, zorder=3, marker="o")
        ax.scatter([mn], [yi], s=34, facecolor="white", edgecolor=fam_col[fam], lw=1.2, zorder=3, marker="o")
        ax.text(1.01, yi, f"−{mc - mn:.1f} dB", va="center", fontsize=6.2, color=fam_col[fam],
                transform=ax.get_yaxis_transform())
    ax.set_yticks(y, [it[1] for it in items])
    # family brackets
    for fam in ("Input", "Linear twin", "Quadratic"):
        ys = [yi for yi, it in zip(y, items) if it[0] == fam]
        ax.axhspan(min(ys) - 0.45, max(ys) + 0.45, color=fam_col[fam], alpha=0.06, lw=0, zorder=0)
        pass
    for t, it in zip(ax.get_yticklabels(), items):
        t.set_color(fam_col[it[0]])
    ax.set_xlabel("Fault-frequency prominence in the envelope spectrum (dB)")
    ax.set_ylim(-0.6, len(items) - 0.4)
    ax.grid(axis="y", visible=False)
    leg = [Line2D([], [], marker="o", ls="", mfc="0.4", mec="k", mew=0.5, ms=5, label="clean"),
           Line2D([], [], marker="o", ls="", mfc="white", mec="0.4", mew=1.2, ms=5, label="0 dB Gaussian noise"),
           Line2D([], [], marker="o", ls="", mfc="0.4", mec="none", alpha=0.4, ms=2.5, label="single channels/seeds")]
    leg += [Rectangle((0, 0), 1, 1, fc=c, alpha=0.5, label=f) for f, c in fam_col.items()]
    ax.legend(handles=leg, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, handletextpad=0.3,
              columnspacing=1.0)
    ax.text(1.01, len(items) - 0.4, "loss at 0 dB", transform=ax.get_yaxis_transform(), fontsize=6,
            color="0.35", va="bottom")
    sns.despine(ax=ax)
    save(fig, out, "Figure_3_prominence")


# ----------------------------------------------------------------------------------------------- fig 4
def fig4_kurtosis(results, out, n=400_000):
    rng = np.random.default_rng(0)
    s, sigma = 1.0, 1.0

    def bg(p, size):  # Bernoulli-Gaussian with unit variance; kurtosis = 3/p
        x = rng.standard_normal(size) * (rng.random(size) < p)
        return x / np.sqrt(p)

    gens = [("Gaussian", lambda k: rng.standard_normal(n), "#2C7FB8"),
            ("Laplace", lambda k: rng.laplace(size=n) / np.sqrt(2), "#2E9E5B"),
            ("Student-$t_5$", lambda k: rng.standard_t(5, n) / np.sqrt(5 / 3), "#9B59B6")]
    pts = []
    for name, g, c in gens:
        w = g(None)
        pts.append((name, c, w))
    for p in (0.3, 0.1, 0.036, 0.01, 0.004):
        pts.append((f"BG p={p}", "#D1495B", bg(p, n)))
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.55), gridspec_kw=dict(width_ratios=[1.05, 1]))
    ax = axes[0]
    kk = np.logspace(np.log10(2.5), np.log10(1000), 200)
    g_var = 4 * s ** 2 * sigma ** 2 + 2 * sigma ** 4
    ax.plot(kk, (4 * s ** 2 * sigma ** 2 + (kk - 1) * sigma ** 4) / g_var, color="#D1495B", lw=1.4,
            label="square law, Proposition 3")
    relu_ref = None
    for name, c, w in pts:
        w = w - w.mean()
        w = w / w.std()
        kap = np.mean(w ** 4)
        mc_sq = np.var((s + w) ** 2)
        mc_re = np.var(np.maximum(s + w, 0))
        if name == "Gaussian":
            relu_ref = mc_re
            g_mc = mc_sq
        ax.scatter([kap], [mc_sq / g_var], s=22, color=c, edgecolor="k", lw=0.4, zorder=3)
        ax.scatter([kap], [mc_re / relu_ref], s=22, marker="s", facecolor="white", edgecolor=c, lw=1.0, zorder=3)
        off = {"Gaussian": (-4, 7), "Laplace": (-6, 9), "Student-$t_5$": (2, 10)}.get(name)
        if off:
            ax.annotate(name, (kap, mc_sq / g_var), xytext=off, textcoords="offset points", fontsize=6,
                        color=c, ha="center")
    ax.axhline(1, color="0.5", lw=0.6, ls=(0, (2, 2)))
    ax.axvline(83.8, color="0.3", lw=0.6, ls=(0, (1, 1)))
    ax.text(95, 1.45, "impulsive noise of\nthe experiments\n($\\kappa_w$ = 83.8)", fontsize=6,
            color="0.3", va="bottom", ha="left")
    ax.text(330, 9, "Bernoulli–Gaussian\n(decreasing\nimpulse rate)", fontsize=6, color="#D1495B", ha="center",
            va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Noise kurtosis $\\kappa_w$ (equal noise power)")
    ax.set_ylabel("Output-noise variance\n(relative to Gaussian noise)")
    leg = [Line2D([], [], color="#D1495B", lw=1.4, label="square law, theory (Prop. 3)"),
           Line2D([], [], marker="o", ls="", mfc="0.5", mec="k", mew=0.4, ms=4.5, label="square law, Monte Carlo"),
           Line2D([], [], marker="s", ls="", mfc="white", mec="0.4", mew=1.0, ms=4.5, label="ReLU, Monte Carlo")]
    ax.legend(handles=leg, loc="upper left")
    ax.set_ylim(0.5, 400)
    panel_label(ax, "a")

    # (b) simulated accuracies
    ax = axes[1]
    acc = defaultdict(list)
    for p in glob.glob(str(results / "sim_mechanism" / "*_seed*" / "metrics.json")):
        m = Path(p).parent.name.rsplit("_seed", 1)[0]
        d = json.loads(Path(p).read_text())
        for k, lab in (("clean", "Clean"), ("white-10", "Gaussian\n−10 dB"), ("impulsive", "Impulsive")):
            if k in d:
                acc[(m, lab)].append(100 * d[k]["acc"])
    if not acc:
        ax.set_visible(False)
    else:
        mods = [("quad", "Quadratic (PhyQ layer)", "#D1495B"), ("noquad", "Linear twin", "#EDA03B"),
                ("classbd", "ClassBD", "#00798C"), ("wdcnn", "WDCNN", "#5D6D7E")]
        conds = ["Clean", "Gaussian\n−10 dB", "Impulsive"]
        w = 0.18
        for i, (m, lab, c) in enumerate(mods):
            xs = np.arange(len(conds)) + (i - 1.5) * w
            mu = [np.mean(acc.get((m, cc), [np.nan])) for cc in conds]
            sd = [np.std(acc.get((m, cc), [np.nan])) for cc in conds]
            ax.bar(xs, mu, w * 0.92, color=c, alpha=0.88, label=lab, lw=0)
            ax.errorbar(xs, mu, sd, fmt="none", ecolor="k", elinewidth=0.6, capsize=1.5)
            for xx, cc in zip(xs, conds):
                vals = acc.get((m, cc), [])
                ax.scatter(np.full(len(vals), xx), vals, s=3, color="k", zorder=3, lw=0)
        ax.set_xticks(np.arange(len(conds)), conds)
        ax.set_ylabel("Accuracy on simulated signals (%)")
        ax.set_ylim(0, 105)
        ax.axhline(25, color="0.5", lw=0.6, ls=(0, (2, 2)))
        ax.grid(axis="x", visible=False)
        ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), columnspacing=0.8, handlelength=1.0)
        panel_label(ax, "b", y=1.13)
    sns.despine(fig)
    fig.tight_layout(w_pad=1.5)
    save(fig, out, "Figure_4_kurtosis")


# ----------------------------------------------------------------------------------------------- fig 5
def fig5_snr(df, out, mode="absolute"):
    d = df[(df.noise == "white") & (df["mode"] == mode)]
    ps = d.groupby(["group", "model", "snr", "seed"]).acc.mean().reset_index()
    st = ps.groupby(["group", "model", "snr"]).acc.agg(["mean", "std"]).reset_index()
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, 4.3), sharex=True)
    order = [m[0] for m in MODELS][::-1]  # proposed drawn last (on top)
    for ax, g, letter in zip(axes.flat, GROUPS, "abcdef"):
        sub = st[st.group == g]
        for m in order:
            s = sub[sub.model == m].sort_values("snr")
            if s.empty:
                continue
            hl = m in (PROPOSED, "classbd")
            lw = 1.9 if m == PROPOSED else (1.2 if hl else 0.8)
            ax.plot(s.snr, s["mean"], color=COLOR[m], ls=LS[m], lw=lw, marker=MARK[m],
                    ms=3.6 if m == PROPOSED else 2.6, mec="white" if m == PROPOSED else COLOR[m], mew=0.4,
                    alpha=1 if hl else 0.85, zorder=5 if m == PROPOSED else (4 if hl else 2))
            if hl:
                ax.fill_between(s.snr, s["mean"] - s["std"].fillna(0), s["mean"] + s["std"].fillna(0),
                                color=COLOR[m], alpha=0.15, lw=0, zorder=1)
        ax.axhline(CHANCE[g], color="0.45", lw=0.6, ls=(0, (1, 1.5)), zorder=0)
        ax.text(10, CHANCE[g], "chance", fontsize=5.8, color="0.45", ha="right", va="bottom")
        ax.set_title(f"{g}  ({NCLS[g]} classes)")
        ax.set_xlim(-10.5, 10.5)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
        lo = max(0, min(CHANCE[g] - 5, np.nanmin(sub["mean"]) - 4))
        ax.set_ylim(lo, 102)
        if ax in axes[:, 0]:
            ax.set_ylabel("Accuracy (%)")
        if ax in axes[1, :]:
            ax.set_xlabel(f"SNR (dB, {mode})")
        panel_label(ax, letter)
    handles = [Line2D([], [], color=COLOR[m], ls=LS[m], marker=MARK[m], lw=1.9 if m == PROPOSED else 1.0,
                      ms=4, label=LABEL[m]) for m, *_ in MODELS]
    handles.append(Rectangle((0, 0), 1, 1, fc="0.6", alpha=0.3, label="± s.d. over seeds"))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), columnspacing=1.2,
               handlelength=2.6)
    sns.despine(fig)
    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    save(fig, out, "Figure_5_snr_curves" if mode == "absolute" else "Figure_S1_snr_relative")


# ----------------------------------------------------------------------------------------------- fig 6
def fig6_heatmap(df, out, snr=-6):
    noises = [("white", "W"), ("pink", "P"), ("impulsive", "I")]
    models = [m[0] for m in MODELS]
    cols, raw, norm = [], [], []
    for g in GROUPS:
        for nz, short in noises:
            st = group_stats(cond(df, nz, "absolute", snr))
            st = st[st.group == g].set_index("model")["mean"]
            v = np.array([st.get(m, np.nan) for m in models])
            raw.append(v)
            norm.append((v - CHANCE[g]) / (100 - CHANCE[g]))
            cols.append((g, short))
    raw, norm = np.array(raw).T, np.array(norm).T
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 0.36 * len(models) + 0.0))
    cmap = sns.color_palette("rocket_r", as_cmap=True)
    sns.heatmap(norm, ax=ax, cmap=cmap, vmin=-0.05, vmax=1.0, cbar=True, linewidths=0.4, linecolor="white",
                annot=np.where(np.isfinite(raw), np.round(raw, 1), np.nan), fmt=".1f",
                annot_kws=dict(fontsize=5.6), cbar_kws=dict(shrink=0.85, pad=0.015, aspect=25))
    # text colour adapts automatically; mark best per column
    for j in range(raw.shape[1]):
        if np.all(np.isnan(raw[:, j])):
            continue
        for i in np.where(raw[:, j] >= np.nanmax(raw[:, j]) - 0.05)[0]:
            ax.add_patch(Rectangle((j + 0.04, i + 0.04), 0.92, 0.92, fill=False, ec="#00B4D8", lw=1.3))
    cb = ax.collections[0].colorbar
    cb.set_label("Accuracy above chance\n(0 = chance, 1 = perfect)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    ax.set_yticks(np.arange(len(models)) + 0.5, [LABEL[m] for m in models], rotation=0)
    for t in ax.get_yticklabels():
        if t.get_text() == LABEL[PROPOSED]:
            t.set_fontweight("bold")
            t.set_color(COLOR[PROPOSED])
    ax.set_xticks(np.arange(len(cols)) + 0.5, [c[1] for c in cols], rotation=0)
    ax.tick_params(axis="x", length=0, pad=2)
    for k, g in enumerate(GROUPS):
        ax.text(3 * k + 1.5, -0.35, g, ha="center", va="bottom", fontsize=7, fontweight="bold")
        if k:
            ax.axvline(3 * k, color="white", lw=2.5)
    n_twins = sum(m in models for m in TWINS)
    ax.axhline(1, color="white", lw=2.0)
    ax.axhline(1 + n_twins, color="white", lw=2.0)
    ax.set_xlabel("W = white (Gaussian), P = pink, I = impulsive (Bernoulli–Gaussian); "
                  f"{snr} dB absolute SNR. Pink and impulsive noise never seen in training.", fontsize=6.3)
    ax.grid(False)
    save(fig, out, "Figure_6_noise_heatmap")


# ----------------------------------------------------------------------------------------------- fig 7
def _holm(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0
    for i, k in enumerate(o):
        run = max(run, (len(ps) - i) * ps[k])
        adj[k] = min(1, run)
    return adj


def fig7_forest(df, results, out):
    from scipy.stats import wilcoxon
    baselines = TWINS + ["classbd", "cnn_transformer", "drsn", "qcnn", "wdcnn", "stats_mlp", "kurtogram_ses"]
    panels = [("CWRU", "CWRU\nWilcoxon, 4 tasks × 3 seeds"), ("PU", "PU (bearing-disjoint)\nWilcoxon, 13 (task, seed) pairs"),
              ("SDUST", "SDUST\nwindow level, 95% CI"), ("IMS", "IMS\nwindow level, 95% CI")]
    extra = ["phyq_nomed_nostats", "phyq_transnet"]  # rest of the Holm family of the text (twins are rows)
    # the width-matched twin (added in v10.10) is NOT part of the Holm family whose p_Holm values are quoted
    # in the text; it is tested with the same paired Wilcoxon test, uncorrected (it is n.s. in every panel)
    no_holm = {"phyq_nomed_noquad_wide"}
    snrs = [(-6, "o", "−6 dB"), (-10, "D", "−10 dB")]
    fig, axes = plt.subplots(1, 4, figsize=(TEXTWIDTH, 3.5), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (g, title), letter in zip(axes, panels, "abcd"):
        for si, (snr, mk, lab) in enumerate(snrs):
            d = cond(df, "white", "absolute", snr)
            if g == "PU":
                d = d[d.group.isin(["PU 5-fold", "PU art2real"])]
            else:
                d = d[d.group == g]
            piv = d.pivot_table(index=["variant", "seed"], columns="model", values="acc")
            res = []
            for b in baselines:
                if b not in piv or PROPOSED not in piv:
                    res.append(None)
                    continue
                pr = piv[[PROPOSED, b]].dropna()
                diff = (pr[PROPOSED] - pr[b]).values
                if g in ("CWRU", "PU"):
                    p = wilcoxon(pr[PROPOSED], pr[b]).pvalue if np.any(diff != 0) else 1.0
                    res.append(dict(mean=diff.mean(), lo=None, hi=None, p=p, pts=diff))
                else:
                    lo, hi = _window_ci(results, g, b, snr, rng)
                    res.append(dict(mean=diff.mean(), lo=lo, hi=hi, p=None, pts=diff))
            if g in ("CWRU", "PU"):
                pe = []
                for b in extra:
                    if b in piv:
                        pr = piv[[PROPOSED, b]].dropna()
                        if len(pr) and np.any(pr[PROPOSED] != pr[b]):
                            pe.append(wilcoxon(pr[PROPOSED], pr[b]).pvalue)
                fam = [i for i, b in enumerate(baselines) if b not in no_holm]
                adj = _holm([res[i]["p"] if res[i] else 1 for i in fam] + pe)[:len(fam)]
                for i, a in zip(fam, adj):
                    if res[i]:
                        res[i]["sig"] = a < 0.05
                for i, b in enumerate(baselines):
                    if b in no_holm and res[i]:
                        res[i]["sig"] = res[i]["p"] < 0.05
            else:
                for r in res:
                    if r:
                        r["sig"] = r["lo"] is not None and (r["lo"] > 0 or r["hi"] < 0)
            for i, (b, r) in enumerate(zip(baselines, res)):
                if r is None:
                    continue
                yy = len(baselines) - 1 - i + (0.17 if si == 0 else -0.17)
                col = (COLOR[PROPOSED] if r["mean"] > 0 else "#00798C") if r["sig"] else "0.6"
                ax.scatter(r["pts"], np.full(len(r["pts"]), yy) + rng.uniform(-0.05, 0.05, len(r["pts"])),
                           s=3, color=col, alpha=0.45, lw=0, zorder=2)
                if r["lo"] is not None:
                    ax.plot([r["lo"], r["hi"]], [yy, yy], color=col, lw=1.4, zorder=3, solid_capstyle="butt")
                ax.scatter([r["mean"]], [yy], marker=mk, s=20, color=col if r["sig"] else "white",
                           edgecolor=col if not r["sig"] else "k", lw=0.6 if r["sig"] else 1.0, zorder=4)
        ax.axvline(0, color="k", lw=0.7)
        ax.axhline(len(baselines) - len(TWINS) - 0.5, color="0.35", lw=0.5, ls=(0, (2, 2)))
        ax.set_title(f"({letter})  " + title, fontsize=6.8, linespacing=1.3)
        ax.grid(axis="y", visible=False)
        lim = max(abs(np.array(ax.get_xlim()))) if ax.has_data() else 10
        ax.set_xlim(-lim, lim)
    axes[0].set_yticks(np.arange(len(baselines))[::-1], [LABEL[b] for b in baselines])
    leg = [Line2D([], [], marker="o", ls="", mfc="0.3", mec="k", ms=4, label="−6 dB"),
           Line2D([], [], marker="D", ls="", mfc="0.3", mec="k", ms=3.6, label="−10 dB"),
           Line2D([], [], marker="s", ls="", color=COLOR[PROPOSED], ms=4, label="PhyQ-TransNet significantly better"),
           Line2D([], [], marker="s", ls="", color="#00798C", ms=4, label="baseline significantly better"),
           Line2D([], [], marker="s", ls="", mfc="white", mec="0.6", ms=4, label="not significant"),
           Line2D([], [], marker="o", ls="", color="0.5", alpha=0.5, ms=2, label="single (task, seed) pairs")]
    fig.legend(handles=leg, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 1.0), columnspacing=1.0,
               handletextpad=0.3)
    sns.despine(fig)
    fig.supxlabel("Accuracy difference, PhyQ-TransNet − baseline (percentage points); white noise, absolute SNR",
                  fontsize=7, y=0.0)
    fig.tight_layout(w_pad=0.6)
    save(fig, out, "Figure_7_forest")


_NPZ_CACHE = {}


def _window_ci(results, g, b, snr, rng, n_boot=2000):
    """grouped (recording) bootstrap CI of the seed-averaged window-level accuracy difference."""
    ds = g.lower()
    key = f"white__absolute__{snr}"
    A, B, rec = [], [], None
    for s in range(3):
        pa = results / "main_tuned" / ds / "chrono" / PROPOSED / f"seed{s}" / "test_probs.npz"
        pb = results / "main_tuned" / ds / "chrono" / b / f"seed{s}" / "test_probs.npz"
        if not (pa.exists() and pb.exists()):
            return None, None
        za, zb = np.load(pa), np.load(pb)
        y = za["y"]
        rec = za["rec_idx"]
        A.append(za[key].argmax(1) == y)
        B.append(zb[key].argmax(1) == y)
    D = (np.mean(A, 0) - np.mean(B, 0)) * 100
    recs, inv = np.unique(rec, return_inverse=True)
    sums = np.bincount(inv, D)
    cnt = np.bincount(inv)
    k = rng.integers(0, len(recs), (n_boot, len(recs)))
    boots = sums[k].sum(1) / cnt[k].sum(1)
    return tuple(np.percentile(boots, [2.5, 97.5]))


# ----------------------------------------------------------------------------------------------- fig 8
def fig8_confusion(df, conf, out, snr=-10.0):
    sets = [("SDUST", "chrono"), ("PU art2real", "art2real")]
    cls = {"SDUST": ['NC', 'IF0.2', 'IF0.4', 'IF0.6', 'OF0.2', 'OF0.4', 'OF0.6', 'RF0.2', 'RF0.4', 'RF0.6'],
           "PU art2real": ["Healthy", "OR", "IR"]}
    exp_of = df.groupby(["group", "model"]).exp.first().to_dict()
    fig = plt.figure(figsize=(TEXTWIDTH, 2.75))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 0.14, 0.52, 0.52, 0.04], wspace=0.12)
    slots = {("SDUST", PROPOSED): 0, ("SDUST", "classbd"): 1, ("PU art2real", PROPOSED): 3,
             ("PU art2real", "classbd"): 4}
    letters = iter("abcd")
    axes_by_set = defaultdict(list)
    for g, var in sets:
        for m in (PROPOSED, "classbd"):
            ax = fig.add_subplot(gs[0, slots[(g, m)]])
            axes_by_set[g].append(ax)
            exp = exp_of.get((g, m))
            M = None
            for s_ in range(3):
                c = conf.get((exp, g, var, m, s_, "white", "absolute", snr))
                if c is not None:
                    M = np.array(c, float) if M is None else M + np.array(c, float)
            if M is None:
                ax.set_visible(False)
                continue
            R = M / M.sum(1, keepdims=True) * 100
            labs = cls[g]
            big = len(labs) > 3
            sns.heatmap(R, ax=ax, cmap="Blues", vmin=0, vmax=100, cbar=False, square=True, linewidths=0.3,
                        linecolor="white", annot=True, fmt=".0f", annot_kws=dict(fontsize=4.4 if big else 6.5))
            ax.set_xticks(np.arange(len(labs)) + 0.5, labs, rotation=90 if big else 0, fontsize=5.3)
            first = slots[(g, m)] in (0, 3)
            ax.set_yticks(np.arange(len(labs)) + 0.5, labs if first else [], rotation=0, fontsize=5.3)
            ax.tick_params(length=0, pad=1.5)
            acc = np.trace(M) / M.sum() * 100
            ax.set_title(f"({next(letters)}) {LABEL[m]}\naccuracy {acc:.1f}%", fontsize=6.6, linespacing=1.25,
                         color=COLOR[m] if m == PROPOSED else "k")
            ax.set_xlabel("Predicted", fontsize=6.2)
            if first:
                ax.set_ylabel("True", fontsize=6.2)
            ax.grid(False)
    fig.canvas.draw()
    ytop = max(a[0].get_position().y1 for a in axes_by_set.values())
    for g, axs in axes_by_set.items():
        b0, b1 = axs[0].get_position(), axs[-1].get_position()
        fig.text((b0.x0 + b1.x1) / 2, ytop + 0.17, f"{g}, white noise {snr:g} dB (3 seeds pooled)",
                 ha="center", fontsize=7.5, fontweight="bold")
    cax = fig.add_subplot(gs[0, 5])
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(0, 100))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Row-normalised (%)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    p0 = axes_by_set["PU art2real"][0].get_position()
    cax.set_position([cax.get_position().x0, p0.y0, cax.get_position().width, p0.height])
    save(fig, out, "Figure_8_confusion")


# ----------------------------------------------------------------------------------------------- fig 9
def fig9_leakage(results, out):
    models = ["phyq_transnet", "cnn_transformer", "drsn", "qcnn", "wdcnn", "stats_mlp"]
    stages = [("seen", "Seen bearings,\nunseen recordings"), ("fold", "Unseen bearings\n(5 folds)"),
              ("a2r", "Artificial → real\ndamage")]
    vals = defaultdict(dict)

    def acc_of(paths):
        per_seed = defaultdict(list)
        for p in paths:
            d = json.loads(Path(p).read_text())
            if "test" in d:
                per_seed[Path(p).parent.name].append(d["test"]["clean"]["acc"])
        if not per_seed:
            return None
        return [np.mean(v) for v in per_seed.values()]

    for m in models:
        vals[m]["seen"] = acc_of(glob.glob(str(results / "pu_protocols" / "pu" / "seen_all" / m / "seed*" / "metrics.json")))
        vals[m]["fold"] = acc_of(glob.glob(str(results / "main_benchmark" / "pu" / "fold*" / m / "seed*" / "metrics.json")))
        vals[m]["a2r"] = acc_of(glob.glob(str(results / "main_benchmark" / "pu" / "art2real" / m / "seed*" / "metrics.json")))
    if not any(v.get("seen") for v in vals.values()):
        print("  skip fig9 (no pu_protocols)")
        return
    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.6, 2.7))
    xs = np.arange(len(stages))
    for m in models:
        mu = [np.mean(vals[m][s]) if vals[m].get(s) else np.nan for s, _ in stages]
        sd = [np.std(vals[m][s]) if vals[m].get(s) else np.nan for s, _ in stages]
        lw = 1.8 if m == "phyq_transnet" else 1.0
        ax.errorbar(xs + (models.index(m) - 2.5) * 0.03, mu, sd, color=COLOR[m], marker=MARK[m], lw=lw, ms=4,
                    capsize=1.5, elinewidth=0.6, label=LABEL[m].replace(" (original)", ""), zorder=3)
    ax.axhline(100 / 3, color="0.45", lw=0.6, ls=(0, (1, 1.5)))
    ax.text(2.25, 100 / 3 + 1, "chance (3 classes)", fontsize=5.8, color="0.45", ha="right", va="bottom")
    ax.axvspan(-0.3, 0.3, color="#D1495B", alpha=0.06, lw=0)
    ax.text(0, 72, "leaky for\nfault diagnosis\n(bearings seen\nin training)", ha="center", fontsize=6,
            color="#D1495B")
    ax.set_xticks(xs, [s[1] for s in stages])
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(25, 102)
    ax.set_ylabel("Clean test accuracy on PU (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), handlelength=1.6)
    sns.despine(ax=ax)
    save(fig, out, "Figure_9_pu_protocols")



# ----------------------------------------------------------------------------------------------- interactive
def interactive_explorer(df, out):
    """Self-contained interactive HTML (plotly) for the supplementary material / GitHub."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    conds = [("white", "absolute", "White noise, absolute SNR"), ("white", "relative", "White noise, relative SNR"),
             ("pink", "absolute", "Pink noise, absolute SNR"), ("impulsive", "absolute", "Impulsive noise, absolute SNR")]
    fig = make_subplots(rows=2, cols=3, subplot_titles=[f"{g} ({NCLS[g]} classes)" for g in GROUPS],
                        horizontal_spacing=0.06, vertical_spacing=0.12)
    vis = []
    for ci, (nz, mode, _) in enumerate(conds):
        d = df[(df.noise == nz) & (df["mode"] == mode)]
        ps = d.groupby(["group", "model", "snr", "seed"]).acc.mean().reset_index()
        st = ps.groupby(["group", "model", "snr"]).acc.agg(["mean", "std"]).reset_index()
        for gi, g in enumerate(GROUPS):
            for m, lab, col, *_ in MODELS:
                sdf = st[(st.group == g) & (st.model == m)].sort_values("snr")
                if sdf.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=sdf.snr, y=sdf["mean"], mode="lines+markers", name=lab, legendgroup=lab,
                    showlegend=(gi == 0), visible=(ci == 0),
                    line=dict(color=col, width=3.5 if m == PROPOSED else 1.6),
                    marker=dict(size=7 if m == PROPOSED else 5),
                    error_y=dict(type="data", array=sdf["std"].fillna(0), thickness=0.8, width=2),
                    hovertemplate=f"<b>{lab}</b><br>{g}<br>SNR %{{x}} dB<br>accuracy %{{y:.1f}}%<extra></extra>"),
                    row=gi // 3 + 1, col=gi % 3 + 1)
                vis.append(ci)
    buttons = [dict(label=title, method="update", args=[{"visible": [v == ci for v in vis]}])
               for ci, (_, _, title) in enumerate(conds)]
    fig.update_layout(template="simple_white", height=760, width=1200,
                      title=dict(text="PhyQ-TransNet: test accuracy vs SNR (mean ± s.d. over seeds)", x=0.01),
                      updatemenus=[dict(buttons=buttons, direction="down", x=0.99, xanchor="right", y=1.12)],
                      legend=dict(orientation="h", y=-0.12), font=dict(family="Arial", size=12))
    fig.update_xaxes(title_text="SNR (dB)", showgrid=True, gridcolor="#eee")
    fig.update_yaxes(title_text="Accuracy (%)", showgrid=True, gridcolor="#eee")
    out.mkdir(parents=True, exist_ok=True)
    fig.write_html(out / "Interactive_results_explorer.html", include_plotlyjs=True, full_html=True)
    print("  wrote", out / "Interactive_results_explorer.html")

# ----------------------------------------------------------------------------------------------- main

# ----------------------------------------------------------------------------------------------- fig 10 / 11
# interpretability on REAL data, from results/<exp>/_analysis/*.npz written by scripts/interpret.py
REP_LABEL = {"input": "Raw input", "input_env": "Hilbert envelope of input (analyst)",
             "stem_pre": "layer output, pre-BN", "stem_post": "post-activation",
             "product": "product term $(W_r x)\\odot(W_g x)$", "power": "power term $W_b(x\\odot x)$"}
COND_LABEL = {"clean": "Clean", "white_0": "White 0 dB", "white_-6": "White −6 dB", "white_-10": "White −10 dB",
              "pink_-6": "Pink −6 dB", "impulsive_-6": "Impulsive −6 dB"}
COND_SHORT = {"clean": "Clean", "white_0": "White\n0 dB", "white_-6": "White\n−6 dB", "white_-10": "White\n−10 dB",
              "pink_-6": "Pink\n−6 dB", "impulsive_-6": "Impulsive\n−6 dB"}
DS_LABEL = {"cwru": "CWRU", "sdust": "SDUST", "pu": "PU", "ims": "IMS", "synthetic": "Synthetic (smoke test)"}
DEPTH_LABEL = {"stem_conv": "stem", "branch1_conv": "branch k31", "branch2_conv": "branch k15",
               "branch3_conv": "branch k7", "fusion_conv": "1×1 fusion", "qres1": "QRes 1", "qres2": "QRes 2",
               "z_d": "$z_d$ (Transformer)"}


def _interp_sets(results: Path, exp: str):
    sets = {}
    for p in sorted((results / exp / "_analysis").glob("*.npz")):
        ds, var, tag = p.stem.split("__")
        sets[(ds, var, tag)] = dict(np.load(p, allow_pickle=False))
    return sets


def _top10(prom):
    ch = np.nanmedian(prom, axis=0)
    ch = ch[np.isfinite(ch)]
    return float(np.sort(ch)[-max(1, len(ch) // 10):].mean()) if len(ch) else np.nan


NULL_RATIOS = (0.83, 0.91, 1.13, 1.21, 1.29)  # off-frequency controls (not harmonics of the fault order)


def _excess_prom(z, rep, cond, band=(15.0, 600.0), n_harm=3):
    """Fault-line prominence of a representation minus the same metric at off-fault control frequencies.

    Computed on the class-averaged power spectra (spec__<rep>__<cond>, shape (K, C, F)) so that spectral
    shape and the max-over-channels selection, which inflate the raw prominence, cancel in the difference.
    Returns (true_db, null_db, excess_db), each the top-10 % channel mean pooled over the fault classes."""
    key = f"spec__{rep}__{cond}"
    if key not in z or "fault_freq_per_class" not in z:
        return np.nan, np.nan, np.nan
    P = z[key].astype(np.float64)
    ff = z["fault_freq_per_class"].astype(float)
    fs, L = float(z["fs"]), int(z["L"])
    f = np.arange(P.shape[-1]) * fs / L                   # bin spacing fs/L for every representation
    df = f[1] - f[0]
    inband = (f >= band[0]) & (f <= band[1])

    def prom(ratio):
        rows = []
        for k, f0 in enumerate(ff):
            if not np.isfinite(f0):
                continue
            harm = np.zeros_like(inband)
            for h in range(1, n_harm + 1):
                harm |= np.abs(f - h * f0 * ratio) <= 1.01 * df
            harm &= inband
            bg = inband & ~harm
            rows.append(10 * np.log10(P[k][:, harm].mean(-1) / (np.median(P[k][:, bg], -1) + 1e-30) + 1e-30))
        return _top10(np.array(rows)) if rows else np.nan

    t = prom(1.0)
    n = float(np.nanmean([prom(r) for r in NULL_RATIOS]))
    return t, n, t - n


def _pick_spectral_dataset(sets):
    dss = sorted({k[0] for k, z in sets.items() if any(f.startswith("prom__") for f in z)})
    return "cwru" if "cwru" in dss else (dss[0] if dss else None)


def fig10_interp_spectra(results, out, exp="interp"):
    sets = _interp_sets(results, exp)
    ds = _pick_spectral_dataset(sets)
    if ds is None:
        print(f"  skip fig10 (no prominence data in results/{exp}/_analysis -> run scripts/interpret.py analyze)")
        return
    Q = {k[1]: z for k, z in sets.items() if k[0] == ds and k[2] == "phyq_nomed"}
    T = {k[1]: z for k, z in sets.items() if k[0] == ds and k[2] == "phyq_nomed_noquad"}
    variants = sorted(Q)
    v0 = variants[0]
    zq = Q[v0]
    fs = float(zq["fs"])
    names = [str(c) for c in zq["class_names"]]

    fig = plt.figure(figsize=(TEXTWIDTH, 5.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.08], width_ratios=[1, 1, 1.15], hspace=0.62, wspace=0.95)

    # ---- (a, b) frequency responses of the 32 stem kernels
    def resp(z):
        nfft = 512
        f = np.fft.rfftfreq(nfft, 1 / fs)
        if bool(z["quad"]):
            Hr = np.abs(np.fft.rfft(z["W_r"], n=nfft, axis=-1))
            Hg = np.abs(np.fft.rfft(z["W_g"], n=nfft, axis=-1))
            R = (Hr * Hg).sum(1)                     # band seen by the bilinear term, summed over input channels
        else:
            R = (np.abs(np.fft.rfft(z["W"], n=nfft, axis=-1)) ** 2).sum(1)
        R = R / R.max(1, keepdims=True)
        order = np.argsort(f[R.argmax(1)])
        return f, 10 * np.log10(R[order] + 1e-6), f[R.argmax(1)]
    peaks = {}
    for j, (lab, z, ttl) in enumerate([("a", zq, "Quadratic stem: $|H_r(f)|\\,|H_g(f)|$"),
                                       ("b", T.get(v0), "Linear twin stem: $|H(f)|^2$")]):
        ax = fig.add_subplot(gs[0, j])
        if z is None:
            ax.axis("off")
            continue
        f, R, pk = resp(z)
        peaks[lab] = pk
        im = ax.imshow(R, aspect="auto", origin="lower", extent=[0, f[-1] / 1e3, 0.5, R.shape[0] + 0.5],
                       cmap="rocket", vmin=-24, vmax=0, interpolation="nearest")
        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel("Kernel (sorted by peak)")
        ax.set_title(ttl, fontsize=7.2, fontweight="normal", pad=4)
        ax.grid(False)
        panel_label(ax, lab, x=-0.42, y=1.04)
        if j == 1:
            cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
            cb.set_label("dB re. kernel max", fontsize=6.5)
            cb.ax.tick_params(labelsize=6)

    # ---- (c) mean input spectrum of the fault classes + where the kernels peak
    ax = fig.add_subplot(gs[0, 2])
    fault = [i for i, c in enumerate(names) if np.isfinite(zq["fault_freq_per_class"][i])]
    healthy = [i for i in range(len(names)) if i not in fault]
    S = zq["spec__input__clean"]                          # (K, C, F)
    fi = np.fft.rfftfreq(2 * (S.shape[-1] - 1), 1 / fs)
    for idx, col, lab in ((fault, "#D1495B", "fault classes"), (healthy, "#5D6D7E", "healthy")):
        if idx:
            s = S[idx].mean((0, 1))
            ax.plot(fi / 1e3, 10 * np.log10(s / s.max() + 1e-12), color=col, lw=0.8, label=lab)
    for lab, col, y0 in (("a", "#D1495B", -60), ("b", "#EDA03B", -65)):
        if lab in peaks:
            ax.plot(peaks[lab] / 1e3, np.full(len(peaks[lab]), y0), "|", color=col, ms=6, mew=0.8)
    ax.text(0.02, 0.17, "kernel peaks: quadratic (top),\nlinear twin (bottom)", transform=ax.transAxes,
            ha="left", fontsize=5.2, color="0.3", bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
    ax.set_ylim(-68, 3)
    ax.set_xlim(0, fs / 2e3)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Input PSD (dB re. max)")
    ax.set_title(f"{DS_LABEL.get(ds, ds)}, clean test windows", fontsize=7.2, fontweight="normal", pad=4)
    ax.legend(loc="upper right", handlelength=1.0, fontsize=5.6, frameon=True, facecolor="white", edgecolor="none",
              framealpha=0.9, borderpad=0.2)
    panel_label(ax, "c", x=-0.42, y=1.04)

    # ---- (d) class-averaged baseband spectra of single feature maps (outer race, clean)
    ax = fig.add_subplot(gs[1, 0:2])
    ffc = zq["fault_freq_per_class"]
    cls = [i for i, c in enumerate(names) if c.startswith("OR")] or fault[:1]
    fo = float(np.nanmean(ffc[cls]))
    curves = [("Raw input", zq, "input", "#5D6D7E"), ("Hilbert envelope of input", zq, "input_env", "#1B1B1B"),
              ("Linear twin stem (pre-activation), best channel", T.get(v0), "stem_pre", "#EDA03B"),
              ("Quadratic product term, best channel", zq, "product", "#D1495B")]
    off = 0
    for lab, z, rep, col in curves:
        if z is None or f"spec__{rep}__clean" not in z:
            continue
        S = z[f"spec__{rep}__clean"][cls].mean(0)          # (C, F)
        pr = z.get(f"prom__{rep}__clean")
        ch = int(np.nanargmax(np.nanmedian(pr[np.isin(z["labels"], cls)], 0))) if pr is not None else 0
        s = S[ch]
        f = np.arange(len(s)) * fs / int(z["L"])           # bin spacing fs/L for every representation
        band = (f >= 15) & (f <= 600)
        db = 10 * np.log10(s / np.median(s[band]) + 1e-12)
        m = f <= 520
        ax.plot(f[m], db[m] + off, color=col, lw=0.9)
        ax.text(515, off + np.nanmax(db[m & (f > 300)]) + 1.5, lab, color=col, fontsize=6.2, va="bottom", ha="right")
        off -= 32
    for k in range(1, 5):
        ax.axvline(k * fo, color="#D1495B", lw=0.5, ls=(0, (3, 2)), alpha=0.7, zorder=0)
        ax.text(k * fo, 1.0, f"{k}×BPFO" if k > 1 else "BPFO", transform=ax.get_xaxis_transform(), fontsize=5.8,
                ha="center", va="bottom", color="#D1495B")
    ax.set_xlim(0, 520)
    ax.set_xticks(np.arange(0, 600, 100))
    ax.set_yticks([])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB re. median, offset)")
    rpm = float(zq["rpm"]) if "rpm" in zq else np.nan
    ax.set_title(f"Outer race, {v0.replace('within_', '')} ({rpm:.0f} rpm): single feature maps, no envelope analysis", fontsize=7.2, fontweight="normal", loc="left", pad=11)
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    panel_label(ax, "d", x=-0.005)

    # ---- (e) excess prominence over off-frequency controls (class-mean spectra), mean over loads
    ax = fig.add_subplot(gs[1, 2])
    items = [("Input", "input: raw", None, "input"), ("Input", "input: Hilbert envelope", None, "input_env"),
             ("Linear", "twin: pre-activation", "T", "stem_pre"), ("Linear", "twin: post-activation", "T", "stem_post"),
             ("Quadratic", "quad.: product term", "Q", "product"), ("Quadratic", "quad.: power term", "Q", "power"),
             ("Quadratic", "quad.: output, pre-BN", "Q", "stem_pre"), ("Quadratic", "quad.: post-activation", "Q", "stem_post")]
    conds = [("clean", "Clean", "0.15"), ("white_0", "White 0 dB", "#56B4E9"), ("impulsive_-6", "Impulsive −6 dB", "#E69F00")]
    fam_col = {"Input": "#5D6D7E", "Linear": "#EDA03B", "Quadratic": "#D1495B"}
    y = np.arange(len(items))[::-1].astype(float)
    h = 0.26
    for ci, (cn, cl, cc) in enumerate(conds):
        for yi, (fam, nm, src, rep) in zip(y, items):
            D = T if src == "T" else Q
            vals = [_excess_prom(D[v], rep, cn)[2] for v in sorted(D)]
            vals = [x for x in vals if np.isfinite(x)]
            if not vals:
                continue
            yy = yi + (1 - ci) * h
            ax.barh(yy, np.nanmean(vals), height=h * 0.92, color=cc, alpha=0.85, lw=0)
            ax.scatter(vals, np.full(len(vals), yy), s=3, color="k", lw=0, zorder=3)
    ax.set_yticks(y, [f"{it[1]}" for it in items])
    for t, it in zip(ax.get_yticklabels(), items):
        t.set_color(fam_col[it[0]])
    ax.axvline(0, color="0.3", lw=0.5)
    ax.set_xlabel("Fault-line prominence above\noff-frequency control (dB)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, fc=cc, label=cl) for _, cl, cc in conds], loc="lower right",
              bbox_to_anchor=(1.02, 1.0), ncol=3, handlelength=0.8, columnspacing=0.6, handletextpad=0.3, fontsize=5.8)
    n_loads = len(variants)
    panel_label(ax, "e", x=-0.80)
    sns.despine(fig)
    save(fig, out, "Figure_10_interp_spectra")


def fig11_interp_fusion(results, out, exp="interp", seed=0):
    sets = _interp_sets(results, exp)
    if not sets:
        print(f"  skip fig11 (no results/{exp}/_analysis/*.npz)")
        return
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    dss = sorted({k[0] for k in sets}, key=lambda d: (d != "cwru", d))
    fig = plt.figure(figsize=(TEXTWIDTH, 5.3))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.05], hspace=0.85)
    top = gs[0].subgridspec(1, 2, width_ratios=[1.35, 1], wspace=0.28)
    bot = gs[1].subgridspec(1, 3, wspace=0.08)
    conds = ["clean", "white_0", "white_-6", "white_-10", "pink_-6", "impulsive_-6"]

    # ---- (a) gate alpha (weight of the statistical stream)
    rows = []
    for (ds, var, tag), z in sets.items():
        for c in conds:
            if f"alpha__{c}" in z:
                a = z[f"alpha__{c}"]
                rows += [{"ds": DS_LABEL.get(ds, ds), "model": LABEL.get(tag, tag), "cond": COND_SHORT[c], "alpha": v}
                         for v in a[np.isfinite(a)]]
    ax = fig.add_subplot(top[0])
    d = pd.DataFrame(rows)
    d["group"] = d["ds"] + " · " + d["model"]
    order = [g for ds in dss for g in (f"{DS_LABEL.get(ds, ds)} · {LABEL['phyq_nomed']}",
                                       f"{DS_LABEL.get(ds, ds)} · {LABEL['phyq_nomed_noquad']}") if g in set(d["group"])]
    pal = {}
    for g in order:
        base = COLOR["phyq_nomed"] if LABEL["phyq_nomed"] in g and "twin" not in g else COLOR["phyq_nomed_noquad"]
        pal[g] = base if g.startswith(DS_LABEL.get(dss[0], dss[0])) else sns.desaturate(base, 0.45)
    sns.boxplot(data=d, x="cond", y="alpha", hue="group", hue_order=order, order=[COND_SHORT[c] for c in conds],
                palette=pal, ax=ax, fliersize=0, linewidth=0.5, width=0.8, whis=(5, 95))
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("")
    ax.set_ylabel("Gate α (weight of statistics stream)")
    ax.tick_params(axis="x", labelsize=6.2)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=6, handlelength=1, columnspacing=0.8)
    panel_label(ax, "a")

    # ---- (b) activation magnitude through depth
    ax = fig.add_subplot(top[1])
    ds0 = dss[0]
    for tag in ("phyq_nomed", "phyq_nomed_noquad"):
        zs = [z for k, z in sets.items() if k[0] == ds0 and k[2] == tag]
        if not zs:
            continue
        layers = [str(s) for s in zs[0]["depth_layers"]]
        for c, col in (("clean", "0.15"), ("white_-6", "#56B4E9"), ("impulsive_-6", "#E69F00")):
            med = np.nanmean([np.nanmedian(z[f"log10rms__{c}"], 0) for z in zs], 0)
            med_c = np.nanmean([np.nanmedian(z["log10rms__clean"], 0) for z in zs], 0)
            ax.plot(np.arange(len(layers)), med - med_c, color=col, ls=LS[tag], marker=MARK[tag], ms=2.6, lw=0.9,
                    mfc=col if tag == "phyq_nomed" else "white")
    ax.set_xticks(np.arange(len(layers)), [DEPTH_LABEL.get(s, s) for s in layers], rotation=40, ha="right",
                  fontsize=6)
    ax.set_ylabel("log$_{10}$ RMS activation re. clean")
    ax.axhline(0, color="0.5", lw=0.5)
    ax.legend(handles=[Line2D([], [], color="0.15", label="clean"),
                       Line2D([], [], color="#56B4E9", label="white −6 dB"),
                       Line2D([], [], color="#E69F00", label="impulsive −6 dB"),
                       Line2D([], [], color="0.4", marker="o", ls="-", ms=3, label=LABEL["phyq_nomed"]),
                       Line2D([], [], color="0.4", marker="s", ls=LS["phyq_nomed_noquad"], mfc="white", ms=3,
                              label=LABEL["phyq_nomed_noquad"])],
              loc="upper left", fontsize=5.8, handlelength=1.6)
    ax.set_title(f"{DS_LABEL.get(ds0, ds0)}, pre-normalisation outputs", fontsize=7, fontweight="normal")
    panel_label(ax, "b")

    # ---- (c) t-SNE of the fused representation z (joint embedding of the three conditions)
    cand = [k for k in sets if k[2] == "phyq_nomed" and "emb_z__clean" in sets[k]]
    cand.sort(key=lambda k: (k[0] != "sdust", -len(sets[k]["emb_labels"])))
    if cand:
        k = cand[0]
        z = sets[k]
        ec = ["clean", "white_-10", "impulsive_-6"]
        X = np.concatenate([z[f"emb_z__{c}"] for c in ec]).astype(np.float64)
        ok = np.isfinite(X).all(1)
        Xs = (X - X[ok].mean(0)) / (X[ok].std(0) + 1e-6)
        Y = np.full((len(X), 2), np.nan)
        Y[ok] = TSNE(2, perplexity=min(30, max(5, ok.sum() // 10)), init="pca", random_state=seed,
                     learning_rate="auto").fit_transform(Xs[ok])
        n = len(z["emb_labels"])
        names = [str(c) for c in z["class_names"]]
        pal = sns.color_palette("tab10" if len(names) <= 10 else "tab20", len(names))
        for j, c in enumerate(ec):
            ax = fig.add_subplot(bot[j])
            ax.scatter(Y[:, 0], Y[:, 1], s=1.2, color="0.88", lw=0, rasterized=True)
            Yi, lab = Y[j * n:(j + 1) * n], z["emb_labels"]
            okj = np.isfinite(Yi).all(1)
            for ci in range(len(names)):
                m = (lab == ci) & okj
                ax.scatter(Yi[m, 0], Yi[m, 1], s=2.2, color=pal[ci], lw=0, rasterized=True,
                           label=names[ci] if j == 0 else None)
            Zc = z[f"emb_z__{c}"].astype(np.float64)
            fin = np.isfinite(Zc).all(1)
            sil = silhouette_score(Zc[fin], lab[fin]) if fin.sum() > 10 and len(np.unique(lab[fin])) > 1 else np.nan
            acc = float(z.get(f"acc__{c}", np.nan))
            ax.set_title(f"{COND_LABEL[c]}\naccuracy {acc:.1f}%,  silhouette {sil:.2f}", fontsize=6.8,
                         fontweight="normal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_linewidth(0.4)
                s.set_color("0.6")
            if j == 0:
                panel_label(ax, "c", x=-0.06)
                ax.set_ylabel(f"t-SNE of fused $z$\n{DS_LABEL.get(k[0], k[0])}, {LABEL['phyq_nomed']}", fontsize=6.5)
            if (~okj).any():
                ax.text(0.02, 0.02, f"{(~okj).sum()} windows non-finite", transform=ax.transAxes, fontsize=5.5,
                        color="0.3")
        fig.legend(*fig.axes[-3].get_legend_handles_labels(), loc="lower center", ncol=min(10, len(names)),
                   bbox_to_anchor=(0.5, 0.03), markerscale=3, fontsize=6, handletextpad=0.1, columnspacing=0.7)
    save(fig, out, "Figure_11_interp_fusion")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of: detectors prominence kurtosis snr heatmap forest confusion leakage interp interactive")
    ap.add_argument("--interp-exp", default="interp", help="experiment folder holding _analysis/ (Fig. 10/11)")
    a = ap.parse_args()
    results = Path(a.results)
    out = Path(a.out) if a.out else results / "_paper_figures"
    setup_style()
    want = set(a.only) if a.only else None
    ok = lambda k: want is None or k in want  # noqa: E731
    df = conf = None
    if any(ok(k) for k in ("snr", "heatmap", "forest", "confusion", "interactive")):
        df, conf = load_runs(results)
        print(f"  loaded {len(df)} rows from main_tuned/main_benchmark")
    if ok("detectors"):
        fig2_detectors(out)
    if ok("prominence"):
        fig3_prominence(results, out)
    if ok("kurtosis"):
        fig4_kurtosis(results, out)
    if ok("snr"):
        fig5_snr(df, out, "absolute")
        fig5_snr(df, out, "relative")
    if ok("heatmap"):
        fig6_heatmap(df, out)
    if ok("forest"):
        fig7_forest(df, results, out)
    if ok("confusion"):
        fig8_confusion(df, conf, out)
    if ok("leakage"):
        fig9_leakage(results, out)
    if ok("interp"):
        fig10_interp_spectra(results, out, a.interp_exp)
        fig11_interp_fusion(results, out, a.interp_exp)
    if ok("interactive"):
        try:
            interactive_explorer(df, out)
        except ImportError:
            print("  skip interactive explorer (pip install plotly)")


if __name__ == "__main__":
    main()
