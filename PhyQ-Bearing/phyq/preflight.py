"""Pre-flight checks run BEFORE any long job: are the dataset folders real and reachable?

Prevents the "hundreds of identical tracebacks" failure mode: one clear message, exit code 1.
Only os.path.isdir / os.listdir are used (read-only).
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List

PLACEHOLDERS = ("D:/Datasets/", "D:/PhyQ_cache", "/path/to/")


def _roots(v) -> List[str]:
    if v is None:
        return []
    return [str(x) for x in (v if isinstance(v, (list, tuple)) else [v])]


def check_data_paths(datasets: Iterable[str]) -> Dict[str, List[str]]:
    """Return {dataset: [problems]} (empty dict = all good)."""
    from .suite import build_run_config
    from .utils.config import interpolate
    problems: Dict[str, List[str]] = {}
    for ds in sorted(set(datasets)):
        if ds == "synthetic":
            continue
        try:
            d = interpolate(build_run_config(ds, "wdcnn", None))["data"]
        except Exception as e:  # missing key in paths.yaml etc.
            problems.setdefault(ds, []).append(f"config error: {e}")
            continue
        cache = os.path.abspath(str(d.get("cache_dir", "cache")))
        for r in _roots(d.get("root")):
            if any(p.lower() in r.replace("\\", "/").lower() for p in PLACEHOLDERS):
                problems.setdefault(ds, []).append(
                    f"'{r}' is the EXAMPLE path -> configs/paths.yaml was not filled with your real folders")
            elif not os.path.isdir(r):
                problems.setdefault(ds, []).append(f"folder not found: '{r}'")
            else:
                ra = os.path.abspath(r)
                same_drive = os.path.splitdrive(cache)[0].lower() == os.path.splitdrive(ra)[0].lower()
                if same_drive and os.path.commonpath([cache, ra]) == ra:
                    problems.setdefault(ds, []).append(f"cache '{cache}' is INSIDE the data folder '{r}'")
    return problems


def report(problems: Dict[str, List[str]]) -> str:
    lines = ["", "!" * 70, "PRE-FLIGHT FAILED - nothing was run, no file was touched:"]
    for ds, ps in problems.items():
        for p in ps:
            lines.append(f"  [{ds}] {p}")
    lines += ["Fix configs/paths.yaml (see docs/RUN_GUIDE_FA.md step 4), then re-run.",
              "Tip: python scripts/check_env.py   shows every path and whether it exists.", "!" * 70]
    return "\n".join(lines)
