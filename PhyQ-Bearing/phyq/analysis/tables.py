"""Aggregate metrics.json files into tidy tables, LaTeX/Markdown and figures."""
from __future__ import annotations

import glob
import json
import os
from typing import List, Optional

import numpy as np
import pandas as pd


def collect(results_dir: str, exp: str) -> pd.DataFrame:
    """One row per (run, condition). condition = 'clean' or (type, mode, snr)."""
    rows = []
    for p in glob.glob(os.path.join(results_dir, exp, "*", "*", "*", "seed*", "metrics.json")):
        m = json.load(open(p, encoding="utf-8"))
        if m.get("val_only"):
            continue
        base = {"dataset": m["dataset"], "variant": m["variant"], "model": m["model_tag"], "seed": m["seed"],
                "params": m["profile"]["params"], "macs": m["profile"].get("macs"),
                "latency_ms": m["profile"].get("latency_ms_bs1"), "best_epoch": m["train"]["best_epoch"],
                "train_time_s": m["train"]["train_time_s"], "path": os.path.dirname(p)}
        for key, res in [("test", m["test"])] + ([("test_tta", m["test_tta"])] if "test_tta" in m else []):
            c = res["clean"]
            rows.append({**base, "eval": key, "noise": "clean", "mode": "clean", "snr": np.nan,
                         "acc": c["acc"], "macro_f1": c["macro_f1"], "bal_acc": c["bal_acc"], "ece": c["ece"],
                         "nll": c["nll"], "n": c["n"]})
            for kind, d1 in res["noise"].items():
                for mode, d2 in d1.items():
                    for snr, r in d2.items():
                        rows.append({**base, "eval": key, "noise": kind, "mode": mode, "snr": float(snr),
                                     "acc": r["acc"], "macro_f1": r["macro_f1"], "bal_acc": r["bal_acc"],
                                     "ece": r["ece"], "nll": r["nll"], "n": r["n"]})
    return pd.DataFrame(rows)


def _fmt(mu, sd, bold=False, latex=False):
    s = f"{mu:.2f} ± {sd:.2f}" if not latex else f"{mu:.2f}$\\pm${sd:.2f}"
    if bold:
        s = f"**{s}**" if not latex else f"\\textbf{{{s}}}"
    return s


def _task(d):
    return d["dataset"] + ":" + d["variant"].str.replace(r"fold\d", "5fold", regex=True)


def main_table(df: pd.DataFrame, metric: str = "acc", mode: str = "absolute", snrs=(None, 0, -6, -10),
               noise: str = "white", latex: bool = False, model_order: Optional[List[str]] = None) -> str:
    """Rows = models; columns = dataset/protocol x condition. Mean ± std over seeds (folds averaged)."""
    d = df[df["eval"] == "test"].copy()
    d["task"] = _task(d)
    cols = [(task, snr) for task in sorted(d["task"].unique()) for snr in snrs]
    models = model_order or sorted(d["model"].unique())
    cells = {}
    for task, snr in cols:
        if snr is None:
            sub = d[(d.task == task) & (d.noise == "clean")]
        else:
            sub = d[(d.task == task) & (d.noise == noise) & (d["mode"] == mode) & (d.snr == float(snr))]
        g = sub.groupby(["model", "seed"])[metric].mean().groupby("model").agg(["mean", "std"]).fillna(0)
        best = g["mean"].max() if len(g) else None
        for mdl in models:
            if mdl in g.index:
                mu, sd = g.loc[mdl, "mean"], g.loc[mdl, "std"]
                cells[(mdl, task, snr)] = _fmt(mu, sd, best is not None and abs(mu - best) < 1e-9, latex)
            else:
                cells[(mdl, task, snr)] = "–"
    hdr = ["Model"] + [f"{t} {'clean' if s is None else f'{s} dB'}" for t, s in cols]
    if latex:
        esc = lambda t: t.replace("_", "\\_")  # noqa: E731
        lines = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
                 " & ".join(esc(h) for h in hdr) + " \\\\", "\\midrule"]
        for mdl in models:
            lines.append(" & ".join([esc(mdl)] + [cells[(mdl, t, s)] for t, s in cols]) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        return "\n".join(lines)
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for mdl in models:
        lines.append("| " + " | ".join([mdl] + [cells[(mdl, t, s)] for t, s in cols]) + " |")
    return "\n".join(lines)


def complexity_table(df: pd.DataFrame) -> str:
    """Complexity per dataset: input length/channels differ between datasets, so a
    single average over datasets (weighted by run count) would be misleading."""
    cols = ["params", "macs", "latency_ms", "train_time_s"]
    by = ["dataset", "model"] if "dataset" in df.columns else ["model"]
    g = df.groupby(by)[cols].mean()
    lines = ["| Dataset | Model | Params | MACs (M) | Latency bs=1 (ms) | Train time / run (min) |",
             "|---|---|---|---|---|---|"]
    for key, r in g.iterrows():
        ds, m = key if isinstance(key, tuple) else ("all", key)
        macs = f"{r['macs'] / 1e6:.1f}" if pd.notna(r["macs"]) else "–"
        lines.append(f"| {ds} | {m} | {int(r['params']):,} | {macs} | {r['latency_ms']:.2f} | {r['train_time_s'] / 60:.1f} |")
    return "\n".join(lines)


def noise_curves(df: pd.DataFrame, out_png: str, metric: str = "acc", mode: str = "absolute", noise: str = "white"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df[(df["eval"] == "test") & (df.noise == noise) & (df["mode"] == mode)].copy()
    if d.empty:
        return
    d["task"] = _task(d)
    tasks = sorted(d.task.unique())
    n = len(tasks)
    ncol = min(n, 4)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False)
    for ax, t in zip(axes.ravel(), tasks):
        sub = d[d.task == t]
        for mdl, g in sub.groupby("model"):
            s = g.groupby(["snr", "seed"])[metric].mean().groupby("snr").agg(["mean", "std"]).fillna(0)
            ax.plot(s.index, s["mean"], marker="o", ms=3, label=mdl)
            ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"], alpha=0.15)
        ax.set_title(t, fontsize=9)
        ax.set_xlabel(f"SNR ({mode}) [dB]")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes.ravel()[n - 1].legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
