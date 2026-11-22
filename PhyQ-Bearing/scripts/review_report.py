"""v10.6 - numbers of the review experiments quoted in the manuscript, recomputed from the released result files.

    python scripts/review_report.py                 # writes results/_paper_assets/review_report.json and prints a summary

Sections (each is skipped if its result folder is missing):
  ablation   results/review_cwru        Table tab:ablation (CWRU, 4 loads x 3 seeds, -6 dB absolute SNR),
                                         paired Wilcoxon test of every configuration against the final model.
  crossload  results/cross_load_tuned   Section sec:crossload (CWRU 12 load pairs x 3 seeds): clean and impulsive
                                         -6 dB accuracy, per source load and per pair, paired Wilcoxon tests.
  interp     results/interp/_analysis   Section sec:interp (CWRU, seed 0): accuracy, excess fault-frequency
                                         prominence (same metric as Fig. 10e), gate alpha, RMS through depth
                                         (same metric as Fig. 11b) and non-finite share.
No number is typed by hand; rounding of the manuscript is ROUND_HALF_UP to one decimal.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

CONDS = ["clean", "white-6", "white-10", "pink-6", "impulsive-6", "harmonic-6", "recorded-6", "recorded-10"]


def _wilcoxon(x, y):
    from scipy.stats import wilcoxon
    d = np.asarray(x, float) - np.asarray(y, float)
    p = float(wilcoxon(x, y).pvalue) if np.any(d != 0) else 1.0
    return {"delta": float(d.mean()), "p": p, "higher": int((d > 0).sum()), "lower": int((d < 0).sum()),
            "n": int(len(d))}


def _load_runs(exp_dir: Path):
    """{model_tag: {(variant, seed): {cond: acc, 'val': best_val, 'impulsive-6_nonfinite': share}}}"""
    A = collections.defaultdict(dict)
    for f in glob.glob(str(exp_dir / "cwru" / "*" / "*" / "seed*" / "metrics.json")):
        d = json.loads(Path(f).read_text().replace("NaN", "null"))
        t = d["test"]
        row = {"clean": t["clean"]["acc"], "val": d.get("train", {}).get("best_val")}
        for nt, modes in t["noise"].items():
            for s, r in modes.get("absolute", {}).items():
                row[f"{nt}{s}"] = r["acc"]
                if "nonfinite" in r:
                    row[f"{nt}{s}_nonfinite"] = r["nonfinite"]
        A[d.get("model_tag", d["model"])][(d["variant"], int(d["seed"]))] = row
    return A


def ablation(results: Path):
    A = _load_runs(results / "review_cwru")
    if "phyq_nomed" not in A:
        return None
    keys = sorted(A["phyq_nomed"])
    out = {"n_runs": len(keys), "models": {}}
    for m, runs in sorted(A.items()):
        common = [k for k in keys if k in runs]
        e = {"n": len(common)}
        for c in CONDS + ["val"]:
            x = [runs[k][c] for k in common if runs[k].get(c) is not None]
            if len(x) != len(common) or not x:
                continue
            e[c] = {"mean": float(np.mean(x))}
            if m != "phyq_nomed":
                e[c].update(_wilcoxon(x, [A["phyq_nomed"][k][c] for k in common]))
        out["models"][m] = e
    return out


def crossload(results: Path):
    A = _load_runs(results / "cross_load_tuned")
    ref, twin = "phyq_nomed", "phyq_nomed_noquad"
    if ref not in A:
        return None
    keys = sorted(A[ref])
    pairs = sorted({k[0] for k in keys})
    out = {"n_runs": len(keys), "pairs": pairs, "models": {}, "vs_proposed": {}}
    for m, runs in sorted(A.items()):
        e = {}
        for c in ("clean", "impulsive-6", "white-6"):
            if not all(c in runs.get(k, {}) for k in keys):
                continue
            e[c] = {"mean": float(np.mean([runs[k][c] for k in keys])),
                    "per_source": {s: float(np.mean([runs[k][c] for k in keys if k[0].startswith(f"cross_{s}to")]))
                                   for s in "0123"},
                    "per_pair": {p: float(np.mean([runs[k][c] for k in keys if k[0] == p])) for p in pairs}}
        out["models"][m] = e
        if m != ref:
            out["vs_proposed"][m] = {c: _wilcoxon([A[ref][k][c] for k in keys], [runs[k][c] for k in keys])
                                     for c in ("clean", "impulsive-6") if all(c in runs.get(k, {}) for k in keys)}
    if twin in A:
        g = {p: out["models"][ref]["clean"]["per_pair"][p] - out["models"][twin]["clean"]["per_pair"][p] for p in pairs}
        out["gain_over_twin_clean_per_pair"] = g
    return out


def interp(results: Path):
    from pro_figures import _excess_prom, _interp_sets
    sets = _interp_sets(results, "interp")
    if not sets:
        return None
    out = {}
    reps = ["input", "input_env", "input_tkeo", "stem_pre", "stem_post", "product", "power"]
    for tag in ("phyq_nomed", "phyq_nomed_noquad"):
        zs = {k[1]: z for k, z in sorted(sets.items()) if k[0] == "cwru" and k[2] == tag}
        if not zs:
            continue
        conds = sorted({f.split("__", 1)[1] for z in zs.values() for f in z if f.startswith("acc__")})
        e = {"loads": list(zs), "acc": {}, "excess_prominence_db": {}, "alpha": {}, "log10rms_re_clean": {},
             "nonfinite_windows": {}}
        for c in conds:
            a = [float(z[f"acc__{c}"]) for z in zs.values()]
            a = [x * 100 if x <= 1.0 else x for x in a]
            e["acc"][c] = {"min": min(a), "max": max(a), "per_load": a}
            al = np.concatenate([z[f"alpha__{c}"] for z in zs.values() if f"alpha__{c}" in z])
            if al.size:
                e["alpha"][c] = {"mean": float(np.nanmean(al)), "median": float(np.nanmedian(al)),
                                 "share_below_0.05": float(np.nanmean(al < 0.05))}
            nf = [float(z[f"nanfrac__{c}"]) * len(z["labels"]) for z in zs.values() if f"nanfrac__{c}" in z]
            e["nonfinite_windows"][c] = {"per_load": nf, "total": float(np.sum(nf)),
                                         "n_windows": int(sum(len(z["labels"]) for z in zs.values()))}
            layers = [str(s) for s in next(iter(zs.values()))["depth_layers"]]
            med = np.nanmean([np.nanmedian(z[f"log10rms__{c}"], 0) for z in zs.values()], 0)
            med_c = np.nanmean([np.nanmedian(z["log10rms__clean"], 0) for z in zs.values()], 0)
            e["log10rms_re_clean"][c] = dict(zip(layers, (med - med_c).astype(float).tolist()))
            if c != "clean":
                share = [np.nanmean(z[f"log10rms__{c}"] - np.nanmedian(z["log10rms__clean"], 0) > 3, 0)
                         for z in zs.values()]
                e["log10rms_re_clean"][c + "__share_windows_gt_3"] = dict(zip(layers, np.mean(share, 0).tolist()))
            for r in reps:
                v = [_excess_prom(z, r, c)[2] for z in zs.values()]
                if np.isfinite(v).any():
                    e["excess_prominence_db"].setdefault(r, {})[c] = {"mean": float(np.nanmean(v)),
                                                                       "per_load": [float(x) for x in v]}
        out[tag] = e
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    results = Path(a.results)
    rep = {}
    for name, fn in (("ablation", ablation), ("crossload", crossload), ("interp", interp)):
        try:
            r = fn(results)
        except FileNotFoundError:
            r = None
        if r is None:
            print(f"skip {name} (no results)")
            continue
        rep[name] = r
    out = Path(a.out) if a.out else results / "_paper_assets" / "review_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    print(f"wrote {out}")
    if "ablation" in rep:
        m = rep["ablation"]["models"]
        print("ablation  I-6: final %.1f, linear twin %.1f (higher in %d/%d, p=%.4f)" % (
            m["phyq_nomed"]["impulsive-6"]["mean"], m["nomed_noquad"]["impulsive-6"]["mean"],
            m["nomed_noquad"]["impulsive-6"]["higher"], m["nomed_noquad"]["impulsive-6"]["n"],
            m["nomed_noquad"]["impulsive-6"]["p"]))
    if "crossload" in rep:
        v = rep["crossload"]["vs_proposed"].get("phyq_nomed_noquad", {})
        for c, s in v.items():
            print(f"crossload {c}: proposed - twin {s['delta']:+.2f} (higher {s['higher']}, lower {s['lower']} "
                  f"of {s['n']}, p={s['p']:.4f})")
    if "interp" in rep:
        q = rep["interp"]["phyq_nomed"]
        print("interp    alpha clean %.2f -> I-6 %.2f; qres2 log10 RMS re. clean (I-6) %.2f" % (
            q["alpha"]["clean"]["mean"], q["alpha"]["impulsive_-6"]["mean"],
            q["log10rms_re_clean"]["impulsive_-6"].get("qres2", np.nan)))


if __name__ == "__main__":
    main()
