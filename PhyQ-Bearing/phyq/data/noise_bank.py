"""Real recorded machine noise for the ``recorded`` test-noise type (v10.5, reviewer request:
"noise recorded on industrial machines").

The bank is built from the HEALTHY experiments of another dataset (default: NREL wind-turbine gearbox,
experiments H*), read-only through the normal dataset adapter, resampled to the sampling rate of the
dataset under test by the signal cache (written into the project cache only), standardised per
experiment and concatenated. Only one channel is used (default: the first one).

Config (``eval.recorded_noise``)::

    eval:
      recorded_noise: {dataset: nrel, label: Healthy, channel: 0}
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np

from .datasets import build_recordings
from .recording import SignalCache


def build_bank(cfg: Dict, target_fs: float, logger=None) -> np.ndarray:
    from ..suite import CFG_DIR
    from ..utils.config import interpolate, load_config
    rcfg = dict(cfg.get("eval", {}).get("recorded_noise") or {})
    src = str(rcfg.get("dataset", "nrel"))
    if src == cfg["data"]["name"]:
        raise ValueError("recorded noise must come from a different machine than the dataset under test")
    scfg = load_config([os.path.join(CFG_DIR, "datasets", f"{src}.yaml")])
    scfg["paths"] = cfg.get("paths", {})
    scfg = interpolate(scfg)
    dcfg = scfg["data"]
    dcfg["cache_dir"] = cfg["data"].get("cache_dir", "cache")
    recs = [r for r in build_recordings(dcfg) if r.label_name == rcfg.get("label", "Healthy")]
    if not recs:
        raise RuntimeError(f"recorded noise: no '{rcfg.get('label', 'Healthy')}' recordings of {src} found")
    cache = SignalCache(dcfg["cache_dir"], float(target_fs))
    ch = int(rcfg.get("channel", 0))
    parts = []
    for r in recs:
        x = np.asarray(cache.get(r)[ch], dtype=np.float32)
        x = x - x.mean()
        parts.append(x / (x.std() + 1e-8))
    bank = np.concatenate(parts)
    if logger:
        logger.info(f"[noise] recorded bank: {src} {len(recs)} x '{rcfg.get('label', 'Healthy')}' "
                    f"recordings, channel {ch}, {len(bank) / target_fs:.1f} s at {target_fs:.0f} Hz")
    return bank
