"""Dataset adapters. Each adapter maps a local folder to a list of ``Recording``.

Adapters only *describe* recordings (lazy loaders); nothing is loaded until the
signal cache materialises it. Add a new dataset by writing ``build(cfg)`` and
registering it below.
"""
from __future__ import annotations

from typing import Callable, Dict, List

from ..recording import Recording
from . import cwru, ims, nrel, pu, sdust, synthetic

REGISTRY: Dict[str, Callable[[dict], List[Recording]]] = {
    "cwru": cwru.build,
    "pu": pu.build,
    "ims": ims.build,
    "nrel": nrel.build,
    "sdust": sdust.build,
    "synthetic": synthetic.build,
}


def build_recordings(dcfg: dict) -> List[Recording]:
    name = dcfg["name"].lower()
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available: {sorted(REGISTRY)}")
    recs = REGISTRY[name](dcfg)
    ids = [r.rid for r in recs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate recording ids")
    return recs
