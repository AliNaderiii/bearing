"""Check that a local dataset is found and parsed correctly BEFORE launching long runs.

    python scripts/sanity_check_data.py cwru pu ims nrel sdust
Prints recordings per class/condition/unit, the split summary of every variant,
and runs the leakage assertions. Also writes results/_data_report/<dataset>.json.
"""
import argparse
import collections
import os
import sys
import traceback

import _bootstrap  # noqa: F401
from _bootstrap import ROOT
from phyq.data.pipeline import prepare
from phyq.preflight import check_data_paths, report as preflight_report
from phyq.data.datasets import build_recordings
from phyq.suite import CFG_DIR, build_run_config
from phyq.utils.config import interpolate, load_yaml
from phyq.utils.misc import save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--variants", nargs="*", help="default: all variants of the dataset")
    a = ap.parse_args()
    problems = check_data_paths(a.datasets)
    if problems:
        print(preflight_report(problems))
        return 1
    failed = []
    for ds in a.datasets:
        print(f"\n################ {ds}")
        try:
            variants = list((load_yaml(os.path.join(CFG_DIR, "datasets", f"{ds}.yaml")).get("split_variants") or {}))
            cfg = interpolate(build_run_config(ds, "wdcnn", None))
            recs = build_recordings(cfg["data"])
            print(f"root = {cfg['data']['root']}  ->  {len(recs)} recordings")
            by = collections.Counter((r.label_name, r.condition) for r in recs)
            for (l, c), n in sorted(by.items()):
                print(f"   {l:10s} {c:14s} {n:5d} recordings")
            print("   units:", sorted({r.group for r in recs})[:40])
            report = {"n_recordings": len(recs), "variants": {}}
            if "_onset_report" in cfg["data"]:
                print("   IMS onset:", cfg["data"]["_onset_report"])
                report["ims_onset"] = cfg["data"]["_onset_report"]
            for v in (a.variants or variants):
                c = interpolate(build_run_config(ds, "wdcnn", v))
                d = prepare(c)
                s = d.summary
                print(f"   [{v}] train/val/test = {s['n_train']}/{s['n_val']}/{s['n_test']} | per-class test "
                      f"{s['test_per_class']} | test units {s['test_groups'][:12]}  -> leakage checks OK")
                print(f"      channels C={d.n_channels}")
                for sk in s.get("skipped_recordings", []):
                    print(f"      [SKIPPED unreadable file] {sk['path']}  ({sk['error'][:120]})")
                report["variants"][v] = s
            save_json(report, os.path.join(ROOT, "results", "_data_report", f"{ds}.json"))
        except Exception:
            traceback.print_exc()
            print(f"!! {ds}: FAILED – fix configs/paths.yaml or configs/datasets/{ds}.yaml")
            failed.append(ds)
    print("\nSANITY: all datasets OK" if not failed else f"\nSANITY FAILED for: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main() or 0)
