"""YAML configuration with inheritance (`extends:`), ``${a.b}`` interpolation and dotted CLI overrides.

Example
-------
>>> cfg = load_config("configs/datasets/cwru.yaml", overrides=["train.epochs=5"])
"""
from __future__ import annotations

import copy
import os
import re
from typing import Any, Dict, Iterable, List, Optional

import yaml


def deep_merge(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``upd`` into a copy of ``base`` (``upd`` wins)."""
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_scalar(s: str) -> Any:
    try:
        return yaml.safe_load(s)
    except yaml.YAMLError:
        return s


def set_dotted(cfg: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def get_dotted(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    d: Any = cfg
    for p in key.split("."):
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    parents = data.pop("extends", None)
    if parents:
        if isinstance(parents, str):
            parents = [parents]
        merged: Dict[str, Any] = {}
        for p in parents:
            ppath = p if os.path.isabs(p) else os.path.normpath(os.path.join(os.path.dirname(path), p))
            merged = deep_merge(merged, load_yaml(ppath))
        data = deep_merge(merged, data)
    return data


def load_config(paths: Iterable[str] | str, overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    """Load and merge one or more YAML files, then apply ``key=value`` overrides."""
    if isinstance(paths, str):
        paths = [paths]
    cfg: Dict[str, Any] = {}
    for p in paths:
        cfg = deep_merge(cfg, load_yaml(p))
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got '{ov}'")
        k, v = ov.split("=", 1)
        set_dotted(cfg, k.strip(), _parse_scalar(v.strip()))
    return cfg


_VAR = re.compile(r"\$\{([^}]+)\}")


def interpolate(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve ``${a.b.c}`` references against the root config (strings only, iterated)."""
    def _res(v, depth=0):
        if isinstance(v, dict):
            return {k: _res(x, depth) for k, x in v.items()}
        if isinstance(v, list):
            return [_res(x, depth) for x in v]
        if isinstance(v, str) and "${" in v and depth < 10:
            full = _VAR.fullmatch(v)
            if full:
                val = get_dotted(cfg, full.group(1))
                if val is None:
                    raise KeyError(f"Unresolved config reference ${{{full.group(1)}}}")
                return _res(val, depth + 1)

            def rep(m):
                val = get_dotted(cfg, m.group(1))
                if val is None:
                    raise KeyError(f"Unresolved config reference ${{{m.group(1)}}}")
                return str(val)
            return _res(_VAR.sub(rep, v), depth + 1)
        return v
    return _res(cfg)


def dump_config(cfg: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
