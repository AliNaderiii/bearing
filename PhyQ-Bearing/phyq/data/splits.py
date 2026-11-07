"""Leakage-safe split protocols.

A split is a list of :class:`Segment` objects per partition (train/val/test).
A segment is a time interval of one recording; windows are cut *inside*
segments only, so no window can straddle two partitions.

Protocols
---------
chronological   : every recording is cut in time: [train | gap | val | gap | test].
                  (CWRU / SDUST within-condition; one recording per class.)
group           : recordings are assigned to partitions by physical unit (``group``)
                  from explicit lists (PU artificial->real, NREL experiment-wise).
group_kfold     : stratified K-fold over physical units (PU bearing-wise folds).
cross_condition : train/val from source conditions (chronological), test on the
                  *test segment* of target conditions -> diagonal == within-condition.
time_blocks     : run-to-failure (IMS): recordings sorted by ``time_order`` per
                  stream and split into contiguous blocks with a gap.
recording       : SEEN units, UNSEEN recordings (PU: 20 separate measurements per
                  bearing x condition). Whole recordings are assigned chronologically
                  per (unit, condition): first 60 % -> train, next 20 % -> val, last
                  20 % -> test. No window/recording is shared between partitions, but
                  the same bearings appear in train and test -> this measures
                  generalisation to new measurements of known bearings (the clean
                  counterpart of the random window splits used by most PU papers),
                  NOT generalisation to new bearings. Optional ``conditions`` filter
                  (single-condition protocols) or ``source``/``target`` lists
                  (cross-condition with recording-disjoint test).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .recording import Recording, SignalCache

PARTS = ("train", "val", "test")


@dataclass
class Segment:
    rec_idx: int
    start: int
    stop: int


def _chrono_segments(n: int, fracs: Sequence[float], gap: int) -> List[tuple]:
    f_tr, f_va, f_te = fracs
    tot = f_tr + f_va + f_te
    f_tr, f_va = f_tr / tot, f_va / tot
    usable = n - 2 * gap
    a = int(usable * f_tr)
    b = int(usable * (f_tr + f_va))
    return [(0, a), (a + gap, b + gap), (b + 2 * gap, n)]


def chronological(recs: List[Recording], cache: SignalCache, fracs=(0.6, 0.2, 0.2),
                  gap: int = 2048, rec_filter: Optional[Sequence[int]] = None) -> Dict[str, List[Segment]]:
    out = {p: [] for p in PARTS}
    idxs = range(len(recs)) if rec_filter is None else rec_filter
    for i in idxs:
        n = cache.length(recs[i])
        for p, (s, e) in zip(PARTS, _chrono_segments(n, fracs, gap)):
            if e > s:
                out[p].append(Segment(i, s, e))
    return out


def group_explicit(recs: List[Recording], cache: SignalCache,
                   assignment: Dict[str, Sequence[str]]) -> Dict[str, List[Segment]]:
    """``assignment = {'train': [...groups], 'val': [...], 'test': [...]}``."""
    lookup = {}
    for p in PARTS:
        for g in assignment.get(p, []) or []:
            if g in lookup:
                raise ValueError(f"Group '{g}' assigned to two partitions")
            lookup[g] = p
    out = {p: [] for p in PARTS}
    for i, r in enumerate(recs):
        p = lookup.get(r.group)
        if p is not None:
            out[p].append(Segment(i, 0, cache.length(r)))
    return out


def group_kfold(recs: List[Recording], cache: SignalCache, n_folds: int, fold: int, seed: int = 0,
                val_from_train: bool = True) -> Dict[str, List[Segment]]:
    """Stratified K-fold over physical units: each unit's recordings go to exactly one partition.

    Test = units of fold ``fold``. Validation = units of fold ``(fold+1) % K``; train = the rest.
    Units are shuffled per label with ``seed``.
    """
    rng = np.random.RandomState(seed)
    groups_by_label: Dict[int, List[str]] = {}
    for r in recs:
        lst = groups_by_label.setdefault(r.label, [])
        if r.group not in lst:
            lst.append(r.group)
    fold_of: Dict[str, int] = {}
    for lbl, gs in groups_by_label.items():
        gs = sorted(gs)
        rng.shuffle(gs)
        for j, g in enumerate(gs):
            fold_of[g] = j % n_folds
    out = {p: [] for p in PARTS}
    for i, r in enumerate(recs):
        f = fold_of[r.group]
        if f == fold:
            p = "test"
        elif val_from_train and f == (fold + 1) % n_folds:
            p = "val"
        else:
            p = "train"
        out[p].append(Segment(i, 0, cache.length(r)))
    return out


def cross_condition(recs: List[Recording], cache: SignalCache, source: Sequence[str], target: Sequence[str],
                    fracs=(0.6, 0.2, 0.2), gap: int = 2048) -> Dict[str, List[Segment]]:
    src = [i for i, r in enumerate(recs) if r.condition in source]
    tgt = [i for i, r in enumerate(recs) if r.condition in target]
    s = chronological(recs, cache, fracs, gap, rec_filter=src)
    t = chronological(recs, cache, fracs, gap, rec_filter=tgt)
    return {"train": s["train"], "val": s["val"], "test": t["test"]}


def time_blocks(recs: List[Recording], cache: SignalCache, fracs=(0.6, 0.2, 0.2), gap_recs: int = 5,
                stream_key=lambda r: (r.group, r.label)) -> Dict[str, List[Segment]]:
    """For run-to-failure snapshot sequences: contiguous chronological blocks per stream."""
    streams: Dict[tuple, List[int]] = {}
    for i, r in enumerate(recs):
        streams.setdefault(stream_key(r), []).append(i)
    out = {p: [] for p in PARTS}
    for _, idx in streams.items():
        idx = sorted(idx, key=lambda i: recs[i].time_order)
        n = len(idx)
        for p, (s, e) in zip(PARTS, _chrono_segments(n, fracs, gap_recs if n > 4 * gap_recs else 0)):
            for i in idx[s:e]:
                out[p].append(Segment(i, 0, cache.length(recs[i])))
    return out


def recording_split(recs: List[Recording], cache: SignalCache, fracs=(0.6, 0.2, 0.2),
                    conditions: Optional[Sequence[str]] = None, source: Optional[Sequence[str]] = None,
                    target: Optional[Sequence[str]] = None) -> Dict[str, List[Segment]]:
    """Whole-recording split per (unit, condition), chronological in ``time_order``.

    * ``conditions``      -> keep only these conditions (train/val/test all from them).
    * ``source/target``   -> train/val from the first 80 % of the SOURCE recordings,
                             test = last 20 % of the TARGET recordings (so a diagonal
                             source == target pair equals the within-condition protocol).
    """
    if (source is None) != (target is None):
        raise ValueError("recording split: give both 'source' and 'target' or neither")
    streams: Dict[tuple, List[int]] = {}
    for i, r in enumerate(recs):
        if conditions and r.condition not in conditions:
            continue
        streams.setdefault((r.group, r.condition), []).append(i)
    out = {p: [] for p in PARTS}
    for (_, cond), idx in sorted(streams.items()):
        idx = sorted(idx, key=lambda i: recs[i].time_order)
        n = len(idx)
        if n < 3:
            raise ValueError(f"recording split needs >= 3 recordings per unit/condition, got {n} "
                             f"for {recs[idx[0]].group}/{cond}")
        f_tr, f_va, f_te = fracs
        tot = f_tr + f_va + f_te
        n_te = max(1, int(round(n * f_te / tot)))
        n_va = max(1, int(round(n * f_va / tot)))
        n_tr = n - n_va - n_te
        if n_tr < 1:
            raise ValueError(f"recording split: not enough recordings for {recs[idx[0]].group}/{cond}")
        parts = {"train": idx[:n_tr], "val": idx[n_tr:n_tr + n_va], "test": idx[n_tr + n_va:]}
        for p in PARTS:
            if source is not None:
                if p in ("train", "val") and cond not in source:
                    continue
                if p == "test" and cond not in target:
                    continue
            for i in parts[p]:
                out[p].append(Segment(i, 0, cache.length(recs[i])))
    return out


def build_split(recs: List[Recording], cache: SignalCache, cfg: dict, window: int) -> Dict[str, List[Segment]]:
    kind = cfg["protocol"]
    gap = int(cfg.get("gap", window))
    fr = tuple(cfg.get("fractions", (0.6, 0.2, 0.2)))
    if kind == "chronological":
        conds = cfg.get("conditions")
        filt = None if not conds else [i for i, r in enumerate(recs) if r.condition in conds]
        return chronological(recs, cache, fr, gap, rec_filter=filt)
    if kind == "group":
        return group_explicit(recs, cache, cfg["assignment"])
    if kind == "group_kfold":
        return group_kfold(recs, cache, int(cfg["n_folds"]), int(cfg["fold"]), int(cfg.get("fold_seed", 0)))
    if kind == "cross_condition":
        return cross_condition(recs, cache, cfg["source"], cfg["target"], fr, gap)
    if kind == "time_blocks":
        return time_blocks(recs, cache, fr, int(cfg.get("gap_recs", 5)))
    if kind == "recording":
        return recording_split(recs, cache, fr, cfg.get("conditions"), cfg.get("source"), cfg.get("target"))
    raise ValueError(f"Unknown split protocol '{kind}'")


def assert_no_leakage(recs: List[Recording], split: Dict[str, List[Segment]], protocol: str) -> None:
    """Hard checks executed before every run."""
    # 1. no time overlap between partitions inside the same recording
    by_rec: Dict[int, List[tuple]] = {}
    for p, segs in split.items():
        for s in segs:
            by_rec.setdefault(s.rec_idx, []).append((s.start, s.stop, p))
    for ri, ivs in by_rec.items():
        ivs.sort()
        for (a0, a1, pa), (b0, b1, pb) in zip(ivs, ivs[1:]):
            if pa != pb and b0 < a1:
                raise AssertionError(f"Time overlap between {pa} and {pb} in recording {recs[ri].rid}")
    # 2. group protocols: a physical unit must not appear in two partitions
    if protocol in ("group", "group_kfold"):
        seen: Dict[str, str] = {}
        for p, segs in split.items():
            for s in segs:
                g = recs[s.rec_idx].group
                if g in seen and seen[g] != p:
                    raise AssertionError(f"Unit '{g}' appears in both {seen[g]} and {p}")
                seen[g] = p
    # 3. recording protocol: a recording belongs to exactly one partition
    if protocol == "recording":
        owner: Dict[int, str] = {}
        for p, segs in split.items():
            for s in segs:
                if s.rec_idx in owner and owner[s.rec_idx] != p:
                    raise AssertionError(f"Recording {recs[s.rec_idx].rid} in both {owner[s.rec_idx]} and {p}")
                owner[s.rec_idx] = p
    # 4. every partition non-empty and train/test share the label set
    for p in PARTS:
        if not split[p]:
            raise AssertionError(f"Partition '{p}' is empty")
    tr = {recs[s.rec_idx].label for s in split["train"]}
    te = {recs[s.rec_idx].label for s in split["test"]}
    if not te.issubset(tr):
        raise AssertionError(f"Test labels {te - tr} never seen in training")
