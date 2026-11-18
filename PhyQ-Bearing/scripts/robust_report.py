"""v10.4 - report for the robustness variants (experiments E1-E3, configs/experiments/robust_*.yaml).

Compares every finished run in results/robust with the PAIRED phyq_nomed / phyq_nomed_noquad / classbd run
(same dataset, split, seed and deterministic test noise) from results/main_tuned (CWRU, IMS, NREL, SDUST) and
results/main_benchmark (PU). Only seeds present for BOTH models are used, so stage-1 (seed 0) and full runs
are both reported correctly. Accuracy in percent, absolute SNR (as in the paper tables).

Also prints the VALIDATION score (best_val, the selection metric: macro-F1 over clean/0/-6 dB white noise) so
it is visible whether a variant costs anything on the protocol's own model-selection criterion.

Outputs: results/robust/robust_report.md and results/robust/robust_report.json
Usage:   python scripts/robust_report.py [--results results]
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from phyq.analysis.stats_tests import mcnemar  # noqa: E402

RES = os.path.join(ROOT, "results")
REF_TAGS = ["phyq_nomed", "phyq_nomed_noquad", "classbd"]
VAR_TAGS = ["phyq_rc", "phyq_rn", "phyq_rc_rn", "phyq_imp", "noquad_imp", "phyq_rc_imp"]
COLS = [("clean", None, None), ("w0", "white", 0), ("w-6", "white", -6), ("w-10", "white", -10),
        ("i0", "impulsive", 0), ("i-6", "impulsive", -6), ("i-10", "impulsive", -10), ("p-6", "pink", -6)]
TASKS = [("CWRU", "cwru", ["within_0HP", "within_1HP", "within_2HP", "within_3HP"]),
         ("IMS", "ims", ["chrono"]), ("NREL", "nrel", ["expwise"]), ("SDUST", "sdust", ["chrono"]),
         ("PU5", "pu", ["fold0", "fold1", "fold2", "fold3", "fold4"]), ("PU-a2r", "pu", ["art2real"])]


def ref_exp(ds):
    return "main_benchmark" if ds == "pu" else "main_tuned"


def run_dir(exp, ds, var, tag, seed):
    return os.path.join(RES, exp, ds, var, tag, f"seed{seed}")


def load_metrics(d):
    p = os.path.join(d, "metrics.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        m = json.load(f)
    return m if "test" in m else None


def acc(m, col):
    name, nt, snr = col
    if nt is None:
        return m["test"]["clean"]["acc"]
    try:
        return m["test"]["noise"][nt]["absolute"][str(snr)]["acc"]
    except KeyError:
        return float("nan")


def probs_key(col):
    name, nt, snr = col
    return "clean" if nt is None else f"{nt}__absolute__{snr}"


def main():
    global RES
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RES, help="results folder (default: <repo>/results)")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="restrict to these seeds (e.g. --seeds 0 for the stage-1 screening table)")
    ap.add_argument("--suffix", default="", help="output name suffix, e.g. _seed0 -> robust_report_seed0.md")
    args = ap.parse_args()
    RES = args.results
    seeds_found = defaultdict(set)
    for p in glob.glob(os.path.join(RES, "robust", "*", "*", "*", "seed*", "metrics.json")):
        parts = os.path.relpath(p, os.path.join(RES, "robust")).replace("\\", "/").split("/")
        ds, var, tag, sd = parts[:4]
        if args.seeds is None or int(sd[4:]) in args.seeds:
            seeds_found[(ds, var, tag)].add(int(sd[4:]))
    if not seeds_found:
        print("No finished runs in results/robust yet.")
        return
    out, lines = {}, []
    lines.append("# Robustness variants (E1-E3) vs paired references\n")
    lines.append("Accuracy %, absolute SNR; mean over the seeds available for BOTH models (n = #runs). "
                 "Delta = variant - phyq_nomed on the same runs. McNemar p pooled over the paired runs (i-6).\n")
    for task, ds, variants in TASKS:
        tags = [t for t in VAR_TAGS if any((ds, v, t) in seeds_found for v in variants)]
        if not tags:
            continue
        lines.append(f"\n## {task}\n")
        hdr = "| model | n | val | " + " | ".join(c[0] for c in COLS) + " | McNemar p (i-6 vs nomed) |"
        lines.append(hdr)
        lines.append("|" + "---|" * (len(COLS) + 4))
        out[task] = {}
        for tag in tags:
            pairs = [(v, s) for v in variants for s in sorted(seeds_found.get((ds, v, tag), set()))]
            rows = {t: [] for t in [tag] + REF_TAGS}
            used = []
            for v, s in pairs:
                mv = load_metrics(run_dir("robust", ds, v, tag, s))
                mr = load_metrics(run_dir(ref_exp(ds), ds, v, "phyq_nomed", s))
                if mv is None or mr is None:
                    continue
                used.append((v, s))
                rows[tag].append(mv)
                for rt in REF_TAGS:
                    m = load_metrics(run_dir(ref_exp(ds), ds, v, rt, s))
                    rows[rt].append(m)
            if not used:
                continue
            out[task][tag] = {}
            # McNemar on impulsive -6 (pooled over the paired runs)
            k = probs_key(COLS[5])
            A, B, Y = [], [], []
            for v, s in used:
                pa = os.path.join(run_dir("robust", ds, v, tag, s), "test_probs.npz")
                pb = os.path.join(run_dir(ref_exp(ds), ds, v, "phyq_nomed", s), "test_probs.npz")
                if os.path.exists(pa) and os.path.exists(pb):
                    za, zb = np.load(pa), np.load(pb)
                    if k in za.files and k in zb.files and len(za["y"]) == len(zb["y"]) and (za["y"] == zb["y"]).all():
                        A.append(za[k].argmax(1)); B.append(zb[k].argmax(1)); Y.append(za["y"])
            pmc = float("nan")
            if Y:
                r = mcnemar(np.concatenate(Y), np.concatenate(A), np.concatenate(B))
                pmc = float(r.get("p", r.get("p_value", float("nan"))))
            out[task][tag] = {"n": len(used), "runs": used, "mcnemar_p_i6": pmc, "vals": {}, "std": {}, "refs": {}}
            for t in [tag] + REF_TAGS:
                ms = [m for m in rows[t] if m is not None]
                if len(ms) != len(used):
                    if t != tag:
                        lines.append(f"| ({t}: missing reference runs) | | | " + " | ".join("" for _ in COLS) + " | |")
                    continue
                vals = [float(np.mean([acc(m, c) for m in ms])) for c in COLS]
                val = float(np.mean([m.get("train", {}).get("best_val", np.nan) for m in ms]))
                rec = dict(zip([c[0] for c in COLS], vals), val=val)
                if t == tag:
                    out[task][tag]["vals"] = rec
                    out[task][tag]["std"] = {c[0]: float(np.std([acc(m, c) for m in ms])) for c in COLS}
                else:
                    out[task][tag]["refs"][t] = rec
                name = f"**{t}**" if t == tag else f"&nbsp;&nbsp;{t}"
                extra = f"{pmc:.1e}" if t == tag and pmc == pmc else ""
                lines.append(f"| {name} | {len(ms)} | {val:.1f} | " + " | ".join(f"{x:.1f}" for x in vals) + f" | {extra} |")
            vv = out[task][tag]["vals"]
            rv = out[task][tag]["refs"].get("phyq_nomed")
            if rv:
                lines.append(f"| &nbsp;&nbsp;Delta vs nomed | | {vv['val'] - rv['val']:+.1f} | " +
                             " | ".join(f"{vv[c[0]] - rv[c[0]]:+.1f}" for c in COLS) + " | |")
    md = "\n".join(lines) + "\n"
    os.makedirs(os.path.join(RES, "robust"), exist_ok=True)
    base = os.path.join(RES, "robust", f"robust_report{args.suffix}")
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(md)
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(md)
    print(f"-> results/robust/robust_report{args.suffix}.md / .json")


if __name__ == "__main__":
    main()
