"""Build paper tables/figures + significance tests from a finished suite.

    python scripts/make_tables.py main_benchmark --reference phyq_transnet
Outputs go to results/<exp>/_report/ (all_runs.csv, tables.md, tables.tex, figures, significance.md).
"""
import argparse
import os

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import ROOT
from phyq.analysis.stats_tests import grouped_bootstrap_ci, holm, mcnemar, wilcoxon_paired
from phyq.analysis.tables import collect, complexity_table, main_table, noise_curves


def significance(df, ref, mode, snr, metric="acc", dataset=None):
    d = df[(df["eval"] == "test")]
    if dataset is not None:
        d = d[d.dataset == dataset]
    if snr is None:
        d = d[d.noise == "clean"]
    else:
        d = d[(d.noise == "white") & (d["mode"] == mode) & (d.snr == float(snr))]
    piv = d.pivot_table(index=["dataset", "variant", "seed"], columns="model", values=metric)
    if ref not in piv.columns:
        return ""          # reference model not run on this dataset -> no block
    rows, ps = [], []
    for m in piv.columns:
        if m == ref:
            continue
        sub = piv[[ref, m]].dropna()
        w = wilcoxon_paired(sub[ref], sub[m])
        rows.append((m, w["mean_diff"], w["p"], w["n"]))
        ps.append(w["p"])
    adj = holm(ps) if ps else []
    out = [f"[{dataset or 'ALL datasets pooled'}] Paired Wilcoxon over (task, seed) pairs, {ref} vs others, condition="
           f"{'clean' if snr is None else f'{mode} {snr} dB'} (Holm-corrected)", "",
           "| vs | mean Δ (pp) | p | p_holm | n |", "|---|---|---|---|---|"]
    for (m, md, p, n), pa in zip(rows, adj):
        out.append(f"| {m} | {md:+.2f} | {p:.4f} | {pa:.4f} | {n} |")
    return "\n".join(out)


def mcnemar_block(df, ref, mode="absolute", snr=-6):
    lines = [f"Exact McNemar per task (seed 0), {ref} vs others, white {mode} {snr} dB; "
             "grouped-bootstrap 95% CI of the reference accuracy (resampling recordings)", "",
             "| task | vs | ref-only correct | other-only correct | p | ref acc [95% CI] |", "|---|---|---|---|---|---|"]
    d = df[(df["eval"] == "test") & (df.noise == "clean") & (df.seed == 0)]
    key = f"white__{mode}__{snr}"
    for (ds, var), g in d.groupby(["dataset", "variant"]):
        paths = dict(zip(g.model, g.path))
        if ref not in paths or not os.path.exists(os.path.join(paths[ref], "test_probs.npz")):
            continue
        zr = np.load(os.path.join(paths[ref], "test_probs.npz"))
        if key not in zr:
            continue
        y, pr = zr["y"], zr[key].argmax(1)
        ci = grouped_bootstrap_ci((pr == y).astype(float), zr["rec_idx"], y)
        for m, p in paths.items():
            if m == ref or not os.path.exists(os.path.join(p, "test_probs.npz")):
                continue
            zo = np.load(os.path.join(p, "test_probs.npz"))
            r = mcnemar(y, pr, zo[key].argmax(1))
            lines.append(f"| {ds}:{var} | {m} | {r['a_only']} | {r['b_only']} | {r['p']:.4f} | "
                         f"{ci['acc']:.2f} [{ci['lo']:.2f}, {ci['hi']:.2f}] |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--metric", default="acc")
    a = ap.parse_args()
    res = os.path.join(ROOT, "results")
    df = collect(res, a.experiment)
    if df.empty:
        print("no results")
        return
    if a.reference is None:   # proposed model: 'phyq_transnet' (main/cross-load) or 'full' (ablation)
        tags = set(df["model"]) if "model" in df else set()
        a.reference = "full" if "full" in tags else "phyq_transnet"
    out = os.path.join(res, a.experiment, "_report")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "all_runs.csv"), index=False)
    md, tex = [], []
    for mode in ("absolute", "relative"):
        if (df["mode"] == mode).any():
            md += [f"## Accuracy (mean ± std over seeds), white noise, {mode} SNR", "",
                   main_table(df, a.metric, mode), ""]
            md += [f"## Macro-F1, white noise, {mode} SNR", "", main_table(df, "macro_f1", mode), ""]
            tex += [f"% {mode} SNR, {a.metric}", main_table(df, a.metric, mode, latex=True), ""]
            noise_curves(df, os.path.join(out, f"noise_curves_{mode}.png"), a.metric, mode)
    for kind in sorted(set(df.noise) - {"clean", "white"}):
        md += [f"## Accuracy, {kind} noise, absolute SNR", "", main_table(df, a.metric, "absolute", noise=kind), ""]
    if (df["eval"] == "test_tta").any():
        md += ["## With test-time augmentation (reported separately)", "",
               main_table(df.assign(eval=df["eval"].replace({"test": "_", "test_tta": "test"})), a.metric), ""]
    md += ["## Complexity", "", complexity_table(df), ""]
    md += ["## Significance (per dataset; pooling datasets would let one dataset dominate)", ""]
    dsets = sorted(set(df.dataset))
    for ds in dsets:
        for mode, snr in (("clean", None), ("absolute", -6), ("absolute", -10), ("relative", -10)):
            blk = significance(df, a.reference, mode, snr, a.metric, dataset=ds)
            if blk:
                md += [blk, ""]
    md += [mcnemar_block(df, a.reference), ""]
    open(os.path.join(out, "tables.md"), "w", encoding="utf-8").write("\n".join(md))
    open(os.path.join(out, "tables.tex"), "w", encoding="utf-8").write("\n".join(tex))
    print("\n".join(md))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
