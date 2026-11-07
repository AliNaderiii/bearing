import numpy as np
import pytest

from phyq.data.datasets import synthetic
from phyq.data.recording import SignalCache
from phyq.data.splits import Segment, assert_no_leakage, build_split, chronological, group_kfold
from phyq.data.windows import make_windows


@pytest.fixture()
def recs_cache(tmp_path):
    recs = synthetic.build({"n_classes": 3, "n_units": 5, "n_conditions": 2, "length": 30000})
    return recs, SignalCache(str(tmp_path))


def test_chronological_disjoint_with_gap(recs_cache):
    recs, cache = recs_cache
    sp = chronological(recs, cache, (0.6, 0.2, 0.2), gap=2048)
    for i in range(len(recs)):
        segs = {p: [s for s in sp[p] if s.rec_idx == i][0] for p in sp}
        assert segs["train"].stop + 2048 <= segs["val"].start
        assert segs["val"].stop + 2048 <= segs["test"].start
    assert_no_leakage(recs, sp, "chronological")


def test_windows_never_cross_segments(recs_cache):
    recs, cache = recs_cache
    sp = chronological(recs, cache, gap=0)
    for p, segs in sp.items():
        w = make_windows(segs, 2048, 256)
        for (ri, st) in w:
            seg = [s for s in segs if s.rec_idx == ri][0]
            assert seg.start <= st and st + 2048 <= seg.stop


def test_group_kfold_unit_disjoint_and_covers(recs_cache):
    recs, cache = recs_cache
    tested = set()
    for f in range(5):
        sp = group_kfold(recs, cache, 5, f, seed=0)
        assert_no_leakage(recs, sp, "group_kfold")
        tested |= {recs[s.rec_idx].group for s in sp["test"]}
    assert tested == {r.group for r in recs}       # every unit tested exactly in one fold


def test_leakage_detector_catches_overlap(recs_cache):
    recs, cache = recs_cache
    bad = {"train": [Segment(0, 0, 10000)], "val": [Segment(0, 9000, 12000)], "test": [Segment(1, 0, 5000)]}
    with pytest.raises(AssertionError):
        assert_no_leakage(recs, bad, "chronological")
    bad2 = {"train": [Segment(0, 0, 100)], "val": [Segment(1, 0, 100)], "test": [Segment(2, 0, 100)]}
    recs2 = [recs[0], recs[1], recs[0]]            # same unit in train and test
    with pytest.raises(AssertionError):
        assert_no_leakage(recs2, bad2, "group")


def test_cross_condition_test_is_target_test_segment(recs_cache):
    recs, cache = recs_cache
    sp = build_split(recs, cache, {"protocol": "cross_condition", "source": ["k0"], "target": ["k1"]}, 2048)
    assert all(recs[s.rec_idx].condition == "k0" for s in sp["train"] + sp["val"])
    assert all(recs[s.rec_idx].condition == "k1" for s in sp["test"])
    n = cache.length(recs[0])
    assert all(s.start > 0.7 * n for s in sp["test"])
