"""v10.4 - LaTeX table for the robustness variants, generated from robust_report*.json (no hand-typed numbers).

Layout: one block of rows per task (validation score, clean, white -10 dB, impulsive -6 dB; absolute SNR),
one column per model. References (phyq_nomed, linear twin, ClassBD) are taken from the SAME runs as the
variants (paired). If variants of one task were evaluated on different run sets, the script stops.

Usage:
  python scripts/robust_table.py --report results/robust/robust_report_seed0.json --kind screen --out tab_remedies_screen.tex
  python scripts/robust_table.py --report results/robust/robust_report.json --kind main --out tab_remedies.tex
"""
from __future__ import annotations

import argparse
import json
import os

KINDS = {
    # stage-1 screening: every pre-specified variant, seed 0
    "screen": ["phyq_nomed", "phyq_rc", "phyq_rn", "phyq_rc_rn", "phyq_imp", "phyq_rc_imp",
               "phyq_nomed_noquad", "noquad_imp", "classbd"],
    # stage-2 full protocol: the variants carried forward
    "main": ["phyq_nomed", "phyq_rc", "phyq_rc_imp", "phyq_nomed_noquad", "classbd"],
}
HEAD = {"phyq_nomed": "PhyQ", "phyq_rc": "+RC", "phyq_rn": "+RN", "phyq_rc_rn": "+RC+RN", "phyq_imp": "+IA",
        "phyq_rc_imp": "+RC+IA", "phyq_nomed_noquad": "Lin.", "noquad_imp": "Lin.+IA", "classbd": "ClassBD"}
REFS = ("phyq_nomed", "phyq_nomed_noquad", "classbd")
ROWS = [("val", "Val."), ("clean", "Clean"), ("w-6", "W $-6$"), ("w-10", "W $-10$"), ("i-6", "I $-6$")]
TASKS = ["CWRU", "IMS", "NREL", "SDUST", "PU5", "PU-a2r"]
TASK_TEX = {"PU5": "PU (5 folds)", "PU-a2r": "PU art2real"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--kind", choices=list(KINDS), default="main")
    ap.add_argument("--out", required=True)
    ap.add_argument("--caption", default=None)
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    rep = json.load(open(a.report, encoding="utf-8"))
    cols = KINDS[a.kind]
    variants = [c for c in cols if c not in REFS]
    body, ns, means = [], {}, {c: {k: [] for k, _ in ROWS} for c in cols}
    for task in TASKS:
        if task not in rep:
            continue
        tv = {v: rep[task][v] for v in variants if v in rep[task]}
        if not tv:
            continue
        runsets = {v: tuple(map(tuple, x["runs"])) for v, x in tv.items()}
        if len(set(runsets.values())) != 1:
            raise SystemExit(f"{task}: variants evaluated on different runs {dict((k, len(r)) for k, r in runsets.items())}"
                             " - restrict with robust_report.py --seeds or finish the missing runs")
        first = next(iter(tv.values()))
        ns[task] = first["n"]
        vals = {}
        for c in cols:
            if c in REFS:
                vals[c] = first["refs"].get(c)
            else:
                vals[c] = tv[c]["vals"] if c in tv else None
        body.append(f"\\multicolumn{{{len(cols) + 1}}}{{l}}{{\\textit{{{TASK_TEX.get(task, task)}}} ($n={first['n']}$ run{'s' if first['n'] > 1 else ''})}}\\\\")
        for key, name in ROWS:
            xs = [vals[c][key] if vals[c] else None for c in cols]
            best = max(x for x in xs if x is not None)
            cells = []
            for c, x in zip(cols, xs):
                if x is None:
                    cells.append("--")
                    continue
                means[c][key].append(x)
                s = f"{x:.1f}"
                cells.append(f"\\textbf{{{s}}}" if abs(x - best) < 0.05 and key != "val" else s)
            body.append(f"\\quad {name} & " + " & ".join(cells) + " \\\\")
        body.append("\\addlinespace")
    # mean over tasks (only if every column is complete)
    body.append(f"\\multicolumn{{{len(cols) + 1}}}{{l}}{{\\textit{{Mean over the {len(ns)} tasks}}}}\\\\")
    for key, name in ROWS:
        xs = [sum(means[c][key]) / len(means[c][key]) if len(means[c][key]) == len(ns) else None for c in cols]
        best = max(x for x in xs if x is not None)
        cells = ["--" if x is None else (f"\\textbf{{{x:.1f}}}" if abs(x - best) < 0.05 and key != "val" else f"{x:.1f}")
                 for x in xs]
        body.append(f"\\quad {name} & " + " & ".join(cells) + " \\\\")
    cap = a.caption or ("Theory-driven variants of PhyQ-TransNet (test accuracy, \\%; absolute SNR; W: white, I: impulsive noise). RC: robust clipping "
                        "after every quadratic convolution (Corollary~\\ref{cor:sat}, $c=9$); RN: median/MAD input "
                        "normalisation; IA: impulsive-noise augmentation with a training-only impact model that differs "
                        "from the test impulses; Lin.: linear twin. Val.: validation score used for model selection "
                        "(macro-F1, clean and white noise only). All models of a block use the same runs and test noise.")
    lab = a.label or ("tab:remedies_screen" if a.kind == "screen" else "tab:remedies")
    tex = ["\\begin{table*}[t]", "\\centering", "\\small" if len(cols) <= 7 else "\\footnotesize",
           f"\\caption{{{cap}}}\\label{{{lab}}}",
           "\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule",
           " & " + " & ".join(HEAD[c] for c in cols) + " \\\\", "\\midrule", *body, "\\bottomrule",
           "\\end{tabular}", "\\end{table*}"]
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("% generated by scripts/robust_table.py from " + os.path.basename(a.report) + " - do not edit by hand\n")
        f.write("\n".join(tex) + "\n")
    print(f"-> {a.out}  ({', '.join(f'{t}: n={n}' for t, n in ns.items())})")


if __name__ == "__main__":
    main()
