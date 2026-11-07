"""Tests for the user's real on-disk layout + the read-only guarantee for dataset folders."""
import os

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio

import mock_data
from phyq.data.datasets import build_recordings
from phyq.data.pipeline import prepare


def _snapshot(root):
    """names + sizes + mtimes of every file: used to prove the data folder is untouched."""
    out = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(dp, f)
            st = os.stat(p)
            out[os.path.relpath(p, root)] = (st.st_size, st.st_mtime_ns)
    return out


def _cfg(root, name, cache, **kw):
    d = {"name": name, "root": root, "cache_dir": cache, "window": 2048, "train_stride": 1024, "eval_stride": 2048}
    d.update(kw)
    return {"data": d, "results_dir": os.path.join(os.path.dirname(cache), "results")}


def test_cwru_list_of_load_folders_and_readonly(tmp_path):
    data = tmp_path / "QCNN" / "data"
    mock_data.make_cwru(str(data / "0HP"), loads=(0,))
    mock_data.make_cwru(str(data / "3HP"), loads=(3,))
    os.makedirs(data / "3HP" / "processed_data_robust", exist_ok=True)
    np.save(data / "3HP" / "processed_data_robust" / "x.npy", np.zeros(3))
    before = _snapshot(data)
    roots = [str(data / "0HP"), str(data / "3HP")]
    cfg = _cfg(roots, "cwru", str(tmp_path / "proj" / "cache"), loads=[0, 3],
               split={"protocol": "cross_condition", "source": ["0HP"], "target": ["3HP"]})
    recs = build_recordings(cfg["data"])
    assert len(recs) == 20
    prepare(cfg)
    assert _snapshot(data) == before
    with pytest.raises(FileNotFoundError, match="1HP"):
        build_recordings(_cfg(roots, "cwru", str(tmp_path / "c"), loads=[1])["data"])


def test_ims_nested_folders(tmp_path):
    root = tmp_path / "4. Bearings" / "IMS"
    mock_data.make_ims(str(root), folders=("1st_test/1st_test", "2nd_test/2nd_test", "4th_test/txt"))
    (root / "1st_test" / "Readme Document for IMS.txt").write_text("doc")
    before = _snapshot(root)
    cfg = _cfg(str(root), "ims", str(tmp_path / "proj" / "cache"),
               onset={"baseline_frac": 0.3, "k_consecutive": 3, "margin_files": 2},
               split={"protocol": "time_blocks", "fractions": [0.6, 0.2, 0.2], "gap_recs": 1})
    d = prepare(cfg)
    assert len(d.class_names) == 4 and d.summary["n_test"] > 0
    assert _snapshot(root) == before


def test_sdust_top_folder_labels_no_channel_duplication(tmp_path):
    root = tmp_path / "SDUST" / "SDUST_bearing"
    rng = np.random.RandomState(0)
    for li, cls in enumerate(["NC", "IF0.2", "IF0.4", "IF0.6", "OF0.2", "OF0.4", "OF0.6", "RF0.2", "RF0.4", "RF0.6"]):
        os.makedirs(root / cls, exist_ok=True)
        n = 30000
        sig = rng.randn(n).astype(np.float32)
        t = np.arange(n) / 25600.0                      # time vector must be ignored
        sio.savemat(root / cls / "data1.mat", {"t": t, "y_values": sig, "copy": sig})  # duplicate ignored
    before = _snapshot(root)
    cfg = _cfg(str(root), "sdust", str(tmp_path / "proj" / "cache"),
               split={"protocol": "chronological", "fractions": [0.6, 0.2, 0.2], "gap": 2048})
    recs = build_recordings(cfg["data"])
    assert sorted(r.label for r in recs) == list(range(10))
    d = prepare(cfg)
    assert d.n_channels == 1                             # real channel count, no fake duplicates
    assert _snapshot(root) == before


def test_nrel_two_big_csvs_staged_once(tmp_path):
    root = tmp_path / "Wind Turbine" / "Data Information"
    os.makedirs(root)
    r = np.random.RandomState(1)
    for fname, pref, ln in (("Healthy_data.csv", "H", 30000), ("Damaged_data.csv", "D", 26000)):
        cols = {}
        for i in range(1, 11):
            for c in (3, 4):
                a = r.randn(30000)
                a[ln - 1000 * (i % 3):] = np.nan        # unequal lengths -> NaN padding at the end
                cols[f"{pref}{i}_AN{c}"] = a
        pd.DataFrame(cols).to_csv(root / fname, index=False)
    before = _snapshot(root)
    cache = tmp_path / "proj" / "cache"
    cfg = _cfg(str(root), "nrel", str(cache), split={"protocol": "group", "assignment": {
        "train": ["D3", "D4", "D5", "D7", "D8", "H1", "H2", "H3", "H4", "H5", "H6", "H8"],
        "val": ["D1", "D2", "D6", "H7"], "test": ["D9", "D10", "H9", "H10"]}})
    d = prepare(cfg)
    assert d.n_channels == 2 and d.summary["n_test"] > 0
    assert len(os.listdir(cache / "nrel" / "_staging")) == 2
    assert _snapshot(root) == before


def test_pu_corrupt_file_skipped_and_reported(tmp_path):
    root = tmp_path / "Paderborn"
    mock_data.make_pu(str(root), files=3, n=16000,
                      bearings=("K001", "K002", "K003", "KA04", "KA15", "KA16", "KI04", "KI14", "KI16"))
    (root / "KA04" / "N15_M07_F10_KA04_3.mat").write_bytes(b"not a mat file")
    os.makedirs(root / "Dataset Information", exist_ok=True)
    before = _snapshot(root)
    cfg = _cfg(str(root), "pu", str(tmp_path / "proj" / "cache"),
               classes={"Healthy": ["K001", "K002", "K003"], "OR": ["KA04", "KA15", "KA16"],
                        "IR": ["KI04", "KI14", "KI16"]},
               conditions=["N15_M07_F10"],
               split={"protocol": "group", "assignment": {
                   "train": ["K001", "KA04", "KI04"], "val": ["K002", "KA15", "KI14"], "test": ["K003", "KA16", "KI16"]}})
    d = prepare(cfg)
    assert [s["rid"] for s in d.summary["skipped_recordings"]] == ["pu_N15_M07_F10_KA04_3"]
    assert _snapshot(root) == before


def test_safety_stop_cache_inside_data(tmp_path):
    mock_data.make_cwru(str(tmp_path / "cwru"), loads=(3,))
    cfg = _cfg(str(tmp_path / "cwru"), "cwru", str(tmp_path / "cwru" / "cache"), loads=[3],
               split={"protocol": "chronological"})
    with pytest.raises(RuntimeError, match="Safety stop"):
        prepare(cfg)
    assert not os.path.exists(tmp_path / "cwru" / "cache")
