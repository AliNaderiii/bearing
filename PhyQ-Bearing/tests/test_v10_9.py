"""v10.9: bearing-disjoint speed shift on PU (group_kfold + source/target) and the capacity-matched linear twin."""
import os

import pytest
import torch

import mock_data
from phyq.data.datasets import build_recordings
from phyq.data.recording import SignalCache
from phyq.data.splits import assert_no_leakage, build_split
from phyq.models.phyq_paper import PhyQTransNet
from phyq.suite import CFG_DIR, build_run_config, expand

CLS = {"Healthy": ["K001", "K002", "K003"], "OR": ["KA04", "KA15", "KA16"], "IR": ["KI04", "KI14", "KI16"]}
CONDS = ("N15_M07_F10", "N09_M07_F10")


@pytest.fixture()
def pu_root(tmp_path):
    mock_data.make_pu(str(tmp_path / "pu"), bearings=sum(CLS.values(), []), conds=CONDS, files=4, n=16000)
    return tmp_path


def _data(root, split):
    return {"name": "pu", "root": str(root / "pu"), "cache_dir": str(root / "cache"), "window": 4096,
            "train_stride": 2048, "eval_stride": 4096, "target_fs": 16000, "classes": CLS,
            "conditions": list(CONDS), "split": split}


def test_group_kfold_speed_shift_is_bearing_disjoint_and_condition_shifted(pu_root):
    base = {"protocol": "group_kfold", "n_folds": 3, "fold": 0, "fold_seed": 0}
    shift = dict(base, source=[CONDS[0]], target=[CONDS[1]])
    d = _data(pu_root, shift)
    recs = build_recordings(d)
    cache = SignalCache(str(pu_root / "c"), 16000)
    sp = build_split(recs, cache, shift, 4096)
    assert_no_leakage(recs, sp, "group_kfold")
    assert all(recs[s.rec_idx].condition == CONDS[0] for s in sp["train"] + sp["val"])
    assert all(recs[s.rec_idx].condition == CONDS[1] for s in sp["test"])
    # same units per partition as the plain fold (fold assignment independent of the conditions)
    plain = build_split(recs, cache, base, 4096)
    for p in ("train", "val", "test"):
        assert {recs[s.rec_idx].group for s in sp[p]} == {recs[s.rec_idx].group for s in plain[p]}
    assert {recs[s.rec_idx].group for s in sp["train"]}.isdisjoint({recs[s.rec_idx].group for s in sp["test"]})
    with pytest.raises(ValueError):
        build_split(recs, cache, dict(base, source=[CONDS[0]]), 4096)


def test_pu_speed_shift_variants_resolve():
    for k in range(5):
        for tgt in ("N09", "N15"):
            cfg = build_run_config("pu", "phyq_paper", f"sp_N15to{tgt}_f{k}")
            sp = cfg["data"]["split"]
            assert sp["protocol"] == "group_kfold" and sp["fold"] == k and sp["source"] == ["N15_M07_F10"]
            assert sp["target"] == [f"{tgt}_M07_F10"]
            assert set(cfg["data"]["conditions"]) == set(sp["source"] + sp["target"])
    specs = list(expand(os.path.join(CFG_DIR, "experiments", "pu_speed_shift.yaml")))
    assert len(specs) == 10 * 2 * 5


def test_width_mult_default_is_original_and_wide_twin_matches_capacity():
    cfg = {"prefilter": {"enabled": False}}
    n = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
    torch.manual_seed(0)
    a = PhyQTransNet(1, 10, dict(cfg, quad=False), 2048)
    torch.manual_seed(0)
    b = PhyQTransNet(1, 10, dict(cfg, quad=False, width_mult=1.0), 2048)
    assert list(a.state_dict()) == list(b.state_dict())
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    q = n(PhyQTransNet(1, 10, dict(cfg), 2048))
    w = PhyQTransNet(1, 10, dict(cfg, quad=False, width_mult=1.65), 2048).eval()
    assert abs(n(w) - q) / q < 0.01
    with torch.no_grad():
        assert w(torch.randn(3, 1, 2048)).shape == (3, 10)
    for f, m in (("capacity_twin.yaml", 36), ("capacity_twin_main.yaml", 21), ("capacity_twin_pu.yaml", 13)):
        s = list(expand(os.path.join(CFG_DIR, "experiments", f)))
        assert len(s) == m and all(x.cfg["model"]["width_mult"] == 1.65 and x.cfg["model"]["quad"] is False for x in s)
