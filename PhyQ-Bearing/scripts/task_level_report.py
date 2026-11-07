"""Task-level (seed-averaged) analysis -- guards against pseudo-replication.

The Wilcoxon in make_tables pairs (task, seed) rows; seeds of the same task share the same test
bearings, so they are not independent. Here every task (fold / protocol) is first averaged over
seeds, then compared. Also prints per-task, per-seed numbers of the reference model and the
val->test gap (is early stopping on held-out bearings informative?).

    python scripts/make_tables.py main_benchmark          # writes all_runs.csv
    python scripts/task_level_report.py main_benchmark [--dataset pu]
Output: results/<exp>/_report/task_level.md
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

import _bootstrap  # noqa: F401
from _bootstrap import ROOT
from phyq.analysis.stats_tests import holm

CONDS = [("clean", "clean", None), ("white", "absolute", -6.0), ("white", "absolute", -10.0),
         ("pink", "absolute", 0.0), ("impulsive", "absolute", 0.0)]


def _sel(d, noise, mode, snr):
    if noise == "clean":
        return d[d.noise == "clean"]
    return d[(d.noise == noise) & (d["mode"] == mode) & (d.snr == snr)]


def _md(df, fmt="{:.2f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join([df.index.name or ""] + [str(c) for c in cols]) + " |",
           "|" + "---|" * (len(cols) + 1)]
    for i, r in df.iterrows():
        out.append("| " + " | ".join([str(i)] + [fmt.format(v) if isinstance(v, (float, np.floating))
                                                  and np.isfinite(v) else str(v) for v in r]) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--reference", default="phyq_transnet")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--metric", default="acc")
    a = ap.parse_args()
    rep = os.path.join(ROOT, "results", a.experiment, "_report")
    df = pd.read_csv(os.path.join(rep, "all_runs.csv"))
    df = df[df["eval"] == "test"]
    if a.dataset:
        df = df[df.dataset == a.dataset]
    df["task"] = df.dataset + ":" + df.variant
    ref = a.reference
    md = [f"# Task-level report ({a.experiment}, metric={a.metric}, reference={ref})", ""]

    # 1) reference model per task x seed + val->test gap
    c = df[df.noise == "clean"]
    r = c[c.model == ref]
    if not r.empty:
        md += ["## Reference model: clean test accuracy per task and seed", "",
               _md(r.pivot_table(index="task", columns="seed", values=a.metric)), ""]
    # best_epoch & val are not in all_runs; show how often each model stops at epoch <= 2
    be = c.groupby("model").best_epoch.apply(lambda s: f"{(s <= 2).mean() * 100:.0f}% (median {s.median():.0f})")
    md += ["## Share of runs whose best epoch is <= 2 (val on held-out bearings uninformative)", "",
           "| model | share (median best epoch) |", "|---|---|"] + [f"| {m} | {v} |" for m, v in be.items()] + [""]

    # 2) task-level tables + tests
    for noise, mode, snr in CONDS:
        d = _sel(df, noise, mode, snr)
        if d.empty:
            continue
        name = "clean" if noise == "clean" else f"{noise} {mode} {snr:g} dB"
        t = d.groupby(["task", "model"])[a.metric].mean().unstack("model")
        if ref not in t.columns:
            continue
        t = t[[ref] + [m for m in t.columns if m != ref]]
        t.loc["MEAN"] = t.mean()
        md += [f"## {name}: seed-averaged per task", "", _md(t), ""]
        t = t.drop("MEAN")
        rows, ps = [], []
        for m in t.columns:
            if m == ref:
                continue
            s = t[[ref, m]].dropna()
            dlt = s[ref] - s[m]
            wins, n = int((dlt > 0).sum()), len(dlt)
            p = float(stats.wilcoxon(s[ref], s[m]).pvalue) if n >= 2 and not np.allclose(dlt, 0) else 1.0
            rows.append((m, dlt.mean(), dlt.median(), wins, n, p))
            ps.append(p)
        adj = holm(ps)
        md += [f"Task-level Wilcoxon (exact, n = tasks; min two-sided p with n=6 is 0.031)", "",
               "| vs | mean Δ | median Δ | wins/tasks | p | p_holm |", "|---|---|---|---|---|---|"]
        for (m, mu, me, w, n, p), pa in zip(rows, adj):
            md.append(f"| {m} | {mu:+.2f} | {me:+.2f} | {w}/{n} | {p:.4f} | {pa:.4f} |")
        md.append("")

    txt = "\n".join(md)
    open(os.path.join(rep, "task_level.md"), "w", encoding="utf-8").write(txt)
    print(txt)
    print(f"\nWritten to {os.path.join(rep, 'task_level.md')}")


if __name__ == "__main__":
    main()
