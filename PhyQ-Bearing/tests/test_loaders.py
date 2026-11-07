import numpy as np
import pytest

import mock_data
from phyq.data.datasets import build_recordings
from phyq.data.pipeline import prepare
from phyq.data.recording import SignalCache


def _cfg(tmp_path, name, **kw):
    d = {"name": name, "root": str(tmp_path / name), "cache_dir": str(tmp_path / "cache"), "window": 2048,
         "train_stride": 1024, "eval_stride": 2048}
    d.update(kw)
    return {"data": d}


def test_cwru_exact_keys_and_no_ba(tmp_path):
    mock_data.make_cwru(str(tmp_path / "cwru"), loads=(2,))
    recs = build_recordings(_cfg(tmp_path, "cwru", loads=[2])["data"])
    assert len(recs) == 10
    r99 = [r for r in recs if r.meta["file_id"] == 99][0]
    assert r99.meta["key_prefix"] == "X099"
    x = r99.loader()
    assert x.shape[0] == 1 and np.abs(x).sum() > 0          # NOT the zero X098 decoy
    with pytest.raises(ValueError):
        build_recordings(_cfg(tmp_path, "cwru", loads=[2], channels=["DE", "BA"])["data"])


def test_cwru_pipeline_within_load(tmp_path):
    mock_data.make_cwru(str(tmp_path / "cwru"), loads=(3,))
    cfg = _cfg(tmp_path, "cwru", loads=[3], split={"protocol": "chronological", "gap": 2048})
    d = prepare(cfg)
    assert d.summary["n_test"] > 0 and len(d.class_names) == 10
    assert d.train[0][0].shape == (1, 2048)


def test_pu_struct_parsing_and_bearing_folds(tmp_path):
    mock_data.make_pu(str(tmp_path / "pu"))
    cls = {"Healthy": ["K001", "K002"], "OR": ["KA04", "KA15"], "IR": ["KI04", "KI14"]}
    cfg = _cfg(tmp_path, "pu", classes=cls, target_fs=16000, split={"protocol": "group", "assignment": {
        "train": ["K001", "KA04", "KI04"], "val": ["K002", "KA15", "KI14"], "test": ["K002", "KA15", "KI14"]}})
    recs = build_recordings(cfg["data"])
    assert len(recs) == 12 and recs[0].loader().shape[0] == 1
    c = SignalCache(str(tmp_path / "c2"), 16000)
    assert c.length(recs[0]) == 16000                      # 64k -> 16k resampling
    with pytest.raises(ValueError):                        # same bearing in val and test -> rejected
        prepare(cfg)


def test_ims_onset_and_time_blocks(tmp_path):
    mock_data.make_ims(str(tmp_path / "ims"))
    cfg = _cfg(tmp_path, "ims", onset={"baseline_frac": 0.3, "k_consecutive": 3, "margin_files": 2},
               split={"protocol": "time_blocks", "gap_recs": 1})
    d = prepare(cfg)
    rep = cfg["data"]["_onset_report"]
    for tag, r in rep.items():
        nf = {"1st_test_B3": 60, "1st_test_B4": 60, "2nd_test_B1": 40, "3rd_test_B3": 50}[tag]
        assert abs(r["onset_idx"] - int(0.8 * nf) - 1) <= 2, (tag, r)
    assert set(d.class_names) == {"Normal", "Inner", "Roller", "Outer"}


def test_nrel_csv_experiment_split(tmp_path):
    mock_data.make_nrel(str(tmp_path / "nrel"))
    cfg = _cfg(tmp_path, "nrel", split={"protocol": "group", "assignment": {
        "train": ["D3", "D4", "D5", "D7", "D8", "H1", "H2", "H3", "H4", "H5", "H6", "H8"],
        "val": ["D1", "D2", "D6", "H7"], "test": ["D9", "D10", "H9", "H10"]}})
    d = prepare(cfg)
    assert d.n_channels == 3 and d.summary["test_groups"] == ["D10", "D9", "H10", "H9"]


def test_sdust_labels_and_conditions(tmp_path):
    mock_data.make_sdust(str(tmp_path / "sdust"))
    recs = build_recordings(_cfg(tmp_path, "sdust")["data"])
    assert sorted({r.label for r in recs}) == list(range(10))
    assert {r.condition for r in recs} == {"1000rpm", "2000rpm"}
    cfg = _cfg(tmp_path, "sdust", split={"protocol": "cross_condition", "source": ["1000rpm"], "target": ["2000rpm"]})
    d = prepare(cfg)
    assert d.n_channels == 3


def test_scale_estimated_on_train_only(tmp_path):
    cfg = {"data": {"name": "synthetic", "root": "none", "cache_dir": str(tmp_path / "c"), "window": 2048,
                    "length": 30000, "split": {"protocol": "chronological"}}}
    d = prepare(cfg)
    x = d.train.materialize()
    assert abs(x.std() - 1.0) < 0.05


def test_pu_long_window_and_optional_bearings(tmp_path):
    """PU protocol v7: 8192-sample windows, group k-fold, optional K006/KI17 skipped if absent."""
    bearings = ("K001", "K002", "K003", "K004", "K005", "KA04", "KA15", "KA16", "KA22", "KA30",
                "KI04", "KI14", "KI16", "KI18", "KI21")                     # no K006 / KI17 on disk
    mock_data.make_pu(str(tmp_path / "pu"), bearings=bearings, files=1, n=64000 * 2)
    cls = {"Healthy": ["K001", "K002", "K003", "K004", "K005", "K006"],
           "OR": ["KA04", "KA15", "KA16", "KA22", "KA30"],
           "IR": ["KI04", "KI14", "KI16", "KI17", "KI18", "KI21"]}
    cfg = _cfg(tmp_path, "pu", classes=cls, target_fs=16000, optional_bearings=["K006", "KI17"],
               window=8192, train_stride=2048, eval_stride=8192,
               split={"protocol": "group_kfold", "n_folds": 5, "fold": 0, "fold_seed": 0})
    recs = build_recordings(cfg["data"])
    assert len(recs) == 15 and {r.group for r in recs}.isdisjoint({"K006", "KI17"})
    d = prepare(cfg)
    assert d.train[0][0].shape == (1, 8192) and len(d.test) > 0
    cfg["data"]["optional_bearings"] = []
    with pytest.raises(FileNotFoundError):
        build_recordings(cfg["data"])
