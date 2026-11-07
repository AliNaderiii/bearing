"""Materialise the resampled signal cache once (optional; run.py does it lazily).

    python scripts/prepare_cache.py cwru pu ims nrel sdust
"""
import sys

import _bootstrap  # noqa: F401
from phyq.data.datasets import build_recordings
from phyq.data.recording import SignalCache
from phyq.suite import build_run_config
from phyq.utils.config import interpolate

for ds in sys.argv[1:]:
    cfg = interpolate(build_run_config(ds, "wdcnn"))
    recs = build_recordings(cfg["data"])
    c = SignalCache(cfg["data"]["cache_dir"], cfg["data"].get("target_fs"))
    for i, r in enumerate(recs, 1):
        c.materialize(r)
        if i % 50 == 0 or i == len(recs):
            print(f"{ds}: {i}/{len(recs)}", flush=True)
