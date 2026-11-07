"""Experiment-suite expansion: (dataset x split variant x model x seed) -> run configs."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

from .utils.config import deep_merge, load_config, load_yaml, set_dotted, _parse_scalar

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_DIR = os.path.join(ROOT, "configs")


@dataclass
class RunSpec:
    exp: str
    dataset: str
    variant: str
    model_tag: str
    seed: int
    cfg: Dict

    @property
    def out_dir(self) -> str:
        base = self.cfg.get("results_dir", "results")
        if not os.path.isabs(base):
            base = os.path.join(ROOT, base)
        return os.path.join(base, self.exp, self.dataset, self.variant, self.model_tag, f"seed{self.seed}")


def _apply(cfg: Dict, overrides: List[str]) -> Dict:
    for ov in overrides or []:
        k, v = ov.split("=", 1)
        set_dotted(cfg, k.strip(), _parse_scalar(v.strip()))
    return cfg


def build_run_config(dataset: str, model: str, variant: Optional[str] = None, overrides: Optional[List[str]] = None,
                     model_overrides: Optional[List[str]] = None) -> Dict:
    cfg = load_config([os.path.join(CFG_DIR, "base.yaml"), os.path.join(CFG_DIR, "datasets", f"{dataset}.yaml"),
                       os.path.join(CFG_DIR, "models", f"{model}.yaml")])
    variants = cfg.pop("split_variants", {}) or {}
    if variant:
        if variant not in variants:
            raise KeyError(f"Variant '{variant}' not defined for {dataset}. Have: {list(variants)}")
        v = dict(variants[variant] or {})
        extra = v.pop("extra_tests", None) or []
        dov = v.pop("data_overrides", None)
        if dov:
            cfg["data"] = deep_merge(cfg["data"], dov)
            if "classes" in dov:
                cfg["data"]["classes"] = dov["classes"]
        if v.get("protocol"):
            cfg["data"]["split"] = v          # replace the default split entirely
    cfg["_variant"] = variant or "default"
    _apply(cfg, model_overrides or [])
    _apply(cfg, overrides or [])
    if variant and extra:
        # Extra test sets evaluated with the SAME trained model (variants whose train/val split is
        # identical, e.g. seen_A and cc_A2B/C/D). The runner verifies the identity before using them.
        cfg["_extra_tests"] = {}
        for name in extra:
            sub = build_run_config(dataset, model, name, overrides, model_overrides)
            sub.pop("_extra_tests", None)
            cfg["_extra_tests"][name] = sub["data"]
    return cfg


def expand(exp_path: str, cli_overrides: Optional[List[str]] = None, only_datasets: Optional[List[str]] = None,
           only_models: Optional[List[str]] = None, seeds: Optional[List[int]] = None) -> Iterator[RunSpec]:
    exp = load_yaml(exp_path)
    name = exp.get("name", os.path.splitext(os.path.basename(exp_path))[0])
    models = exp["models"]
    for run in exp["runs"]:
        ds = run["dataset"]
        if only_datasets and ds not in only_datasets:
            continue
        for var in run.get("variants", [None]):
            for m in models:
                if isinstance(m, str):
                    m = {"tag": m, "config": m}
                if only_models and m["tag"] not in only_models:
                    continue
                # CLI --seeds > per-run seeds > experiment seeds
                for s in (seeds or run.get("seeds") or exp.get("seeds", [0])):
                    cfg = build_run_config(ds, m["config"], var,
                                           list(exp.get("overrides", [])) + list(run.get("overrides", [])) +
                                           list(cli_overrides or []),
                                           m.get("overrides"))
                    cfg["_model_tag"] = m["tag"]
                    cfg["_val_only"] = bool(exp.get("val_only", False))
                    yield RunSpec(name, ds, var or "default", m["tag"], int(s), cfg)
