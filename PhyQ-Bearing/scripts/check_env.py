"""Step-1 check: Python packages, GPU, and config paths.

    python scripts/check_env.py
"""
import importlib
import os
import sys

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

ok = True
print(f"Python {sys.version.split()[0]}")
for pkg in ["torch", "numpy", "scipy", "pandas", "sklearn", "yaml", "matplotlib", "thop", "pytest"]:
    try:
        m = importlib.import_module(pkg)
        print(f"  [OK] {pkg} {getattr(m, '__version__', '')}")
    except Exception as e:
        ok = False
        print(f"  [MISSING] {pkg}: {e}")
try:
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  [OK] CUDA GPU: {p.name}, {p.total_memory / 2**30:.1f} GB, torch CUDA {torch.version.cuda}")
        x = torch.randn(64, 1, 2048, device="cuda")
        with torch.autocast("cuda"):
            torch.nn.Conv1d(1, 8, 7).cuda()(x)
        print("  [OK] GPU forward pass + AMP")
    else:
        ok = False
        print("  [WARN] CUDA NOT available -> install the CUDA build of PyTorch (see guide step 3)")
except Exception as e:
    ok = False
    print(f"  [ERROR] torch: {e}")

if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
    print(f"  [WARNING] not inside a virtual environment ({sys.executable}) -> run .\\.venv\\Scripts\\Activate.ps1")

paths_file = os.path.join(ROOT, "configs", "paths.yaml")
if not os.path.exists(paths_file):
    ok = False
    print("  [MISSING] configs/paths.yaml -> copy configs/paths.example.yaml and edit it")
else:
    import yaml
    paths = (yaml.safe_load(open(paths_file, encoding="utf-8")) or {}).get("paths", {})
    for k, v in paths.items():
        if k == "cache":
            print(f"  [OK] cache -> {v}")
            continue
        for vv in (v if isinstance(v, list) else [v]):
            exists = os.path.isdir(vv)
            ok = ok and exists
            print(f"  [{'OK' if exists else 'NOT FOUND'}] {k:6s} -> {vv}")
    for k, v in paths.items():
        if k != "cache":
            for vv in (v if isinstance(v, list) else [v]):
                c = os.path.abspath(paths.get("cache", "cache"))
                if os.path.splitdrive(c)[0] == os.path.splitdrive(os.path.abspath(vv))[0] and \
                        os.path.commonpath([c, os.path.abspath(vv)]) == os.path.abspath(vv):
                    ok = False
                    print(f"  [ERROR] cache folder is inside the {k} data folder -> move it (data must stay untouched)")
    try:
        from phyq.preflight import check_data_paths, report
        probs = check_data_paths([k for k in paths if k != "cache"])
        if probs:
            ok = False
            print(report(probs))
    except Exception as e:
        ok = False
        print(f"  [ERROR] preflight: {e}")
print("\nALL GOOD" if ok else "\nFix the items above before continuing")
sys.exit(0 if ok else 1)
