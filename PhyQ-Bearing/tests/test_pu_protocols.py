"""v9: PU seen-bearing (recording-disjoint) and cross-condition protocols."""
import pytest

import mock_data
from phyq.data.datasets import build_recordings
from phyq.data.pipeline import prepare
from phyq.data.recording import SignalCache
from phyq.data.splits import Segment, assert_no_leakage, build_split
from phyq.suite import build_run_config, expand

CLS = {"Healthy": ["K001", "K002"], "OR": ["KA04", "KA15"], "IR": ["KI04", "KI14"]}
CONDS = ("N15_M07_F10", "N09_M07_F10")


def _cfg(tmp_path, split, **kw):
    d = {"name": "pu", "root": str(tmp_path / "pu"), "cache_dir": str(tmp_path / "cache"), "window": 4096,
         "train_stride": 2048, "eval_stride": 4096, "target_fs": 16000, "classes": CLS,
         "conditions": list(CONDS), "split": split}
    d.update(kw)
    return {"data": d}


@pytest.fixture()
def pu_root(tmp_path):
    mock_data.make_pu(str(tmp_path / "pu"), bearings=sum(CLS.values(), []), conds=CONDS, files=10, n=32000)
    return tmp_path


def test_recording_split_disjoint_chronological(pu_root):
    cfg = _cfg(pu_root, {"protocol": "recording", "fractions": [0.6, 0.2, 0.2]})
    recs = build_recordings(cfg["data"])
    cache = SignalCache(str(pu_root / "c"), 16000)
    sp = build_split(recs, cache, cfg["data"]["split"], 4096)
    assert_no_leakage(recs, sp, "recording")
    for g in {r.group for r in recs}:
        for c in CONDS:
            ks = {p: sorted(recs[s.rec_idx].time_order for s in sp[p]
                            if recs[s.rec_idx].group == g and recs[s.rec_idx].condition == c) for p in sp}
            assert ks["train"] == list(range(1, 7)) and ks["val"] == [7, 8] and ks["test"] == [9, 10]
    # seen-bearing protocol: every bearing is in train AND test, but no recording is shared
    tr = {s.rec_idx for s in sp["train"]}
    te = {s.rec_idx for s in sp["test"]}
    assert tr.isdisjoint(te)
    assert {recs[i].group for i in tr} == {recs[i].group for i in te}


def test_recording_cross_condition(pu_root):
    split = {"protocol": "recording", "source": [CONDS[0]], "target": [CONDS[1]]}
    cfg = _cfg(pu_root, split)
    recs = build_recordings(cfg["data"])
    sp = build_split(recs, SignalCache(str(pu_root / "c"), 16000), split, 4096)
    assert_no_leakage(recs, sp, "recording")
    assert all(recs[s.rec_idx].condition == CONDS[0] for s in sp["train"] + sp["val"])
    assert all(recs[s.rec_idx].condition == CONDS[1] and recs[s.rec_idx].time_order >= 9 for s in sp["test"])
    with pytest.raises(ValueError):
        build_split(recs, SignalCache(str(pu_root / "c"), 16000), {"protocol": "recording", "source": [CONDS[0]]},
                    4096)


def test_recording_leakage_detector(pu_root):
    cfg = _cfg(pu_root, {"protocol": "recording"})
    recs = build_recordings(cfg["data"])
    bad = {"train": [Segment(0, 0, 100)], "val": [Segment(1, 0, 100)], "test": [Segment(0, 200, 300)]}
    with pytest.raises(AssertionError):
        assert_no_leakage(recs, bad, "recording")


def test_recording_pipeline_end_to_end(pu_root):
    d = prepare(_cfg(pu_root, {"protocol": "recording"}, conditions=[CONDS[0]]))
    s = d.summary
    assert s["protocol"] == "recording" and s["n_recordings"] == 60
    assert s["train_groups"] == s["test_groups"] == sorted(sum(CLS.values(), []))
    assert len(d.train) > 0 and len(d.val) > 0 and len(d.test) > 0


def test_pu_yaml_variants_resolve():
    for v in ["seen_all", "seen_A", "seen_B", "seen_C", "seen_D", "paper_dong7"] + [f"cc_{a}2{b}" for a in "ABCD" for b in "ABCD" if a != b]:
        cfg = build_run_config("pu", "phyq_paper", v)
        sp = cfg["data"]["split"]
        assert sp["protocol"] == "recording"
        if v.startswith("cc_"):
            assert cfg["data"]["conditions"] == sp["source"] + sp["target"]
            assert sp["source"] != sp["target"]
    c7 = build_run_config("pu", "wdcnn", "paper_dong7")["data"]
    assert len(c7["classes"]) == 7 and c7["conditions"] == ["N15_M01_F10"]
    # the default (headline) protocol is still bearing-wise
    assert build_run_config("pu", "wdcnn")["data"]["split"]["protocol"] == "group_kfold"


def test_pu_protocol_suites_expand():
    import os
    from phyq.suite import CFG_DIR
    specs = list(expand(os.path.join(CFG_DIR, "experiments", "pu_protocols.yaml")))
    assert len(specs) == 6 * 3 * 6
    q = list(expand(os.path.join(CFG_DIR, "experiments", "pu_quick.yaml")))
    assert len(q) == 3 and all(s.model_tag == "phyq_transnet" for s in q)
    w = list(expand(os.path.join(CFG_DIR, "experiments", "pu_quick_w2048.yaml")))
    assert len(w) == 6 and all(s.cfg["data"]["window"] == 2048 for s in w)
    # the overrides reach the extra test sets too (identical windows are required)
    assert all(d["window"] == 2048 for s in w for d in (s.cfg.get("_extra_tests") or {}).values())
    # seen_X carries the 3 cross-condition test sets of the same source condition
    ex = build_run_config("pu", "wdcnn", "seen_B")["_extra_tests"]
    assert sorted(ex) == ["cc_B2A", "cc_B2C", "cc_B2D"]
    assert ex["cc_B2A"]["split"]["source"] == ["N09_M07_F10"] and ex["cc_B2A"]["split"]["target"] == ["N15_M07_F10"]


def test_cache_length_keeps_no_open_files(pu_root):
    """v9 fix: 1360 PU recordings -> 'Too many open files' on Linux when every length() kept a memmap."""
    cfg = _cfg(pu_root, {"protocol": "recording"})
    recs = build_recordings(cfg["data"])
    c = SignalCache(str(pu_root / "c3"), 16000)
    for r in recs:
        assert c.length(r) == 8000
    assert c._mm == {} and c.shape(recs[0]) == (1, 8000)


def test_memmap_dataset_bounded_open_files(pu_root):
    from phyq.data.torch_dataset import WindowDataset
    c = SignalCache(str(pu_root / "c4"), 16000)
    recs = build_recordings(_cfg(pu_root, {"protocol": "recording"})["data"])
    paths = [c.materialize(r) for r in recs]
    idx = [(i, 0) for i in range(len(paths))]
    ds = WindowDataset(paths, idx, [0] * len(idx), 4096, None, None, False, {})
    ds.MAX_OPEN_MEMMAPS = 16
    for i in range(len(ds)):
        ds[i]
    assert len(ds._arrs) <= 16


def test_extra_tests_reuse_model_only_if_split_identical(pu_root, tmp_path):
    """seen_X model is evaluated on cc_X2Y test sets -> identical train/val windows are verified."""
    from phyq.runner import run_experiment
    src = {"protocol": "recording", "fractions": [0.6, 0.2, 0.2]}
    base = _cfg(pu_root, src, conditions=[CONDS[0]])["data"]
    cc = _cfg(pu_root, dict(src, source=[CONDS[0]], target=[CONDS[1]]))["data"]
    bad = _cfg(pu_root, dict(src, source=[CONDS[1]], target=[CONDS[0]]))["data"]   # other source
    cfg = build_run_config("pu", "stats_mlp")
    cfg["data"] = dict(cfg["data"], **base)
    cfg["data"].pop("optional_bearings", None)
    cfg.update(device="cpu", _variant="seen_X", _model_tag="stats_mlp", results_dir=str(tmp_path / "res"),
               _extra_tests={"cc_X2Y": dict(cfg["data"], **cc), "cc_bad": dict(cfg["data"], **bad)})
    cfg["train"].update(epochs=1, min_epochs=1, batch_size=8, amp=False)
    cfg["selection"].update(batch_size=16, snrs=["clean"])
    cfg["eval"].update(batch_size=16, snrs=[0], noise_types=["white"])
    out_dir = tmp_path / "res" / "exp" / "pu" / "seen_X" / "stats_mlp" / "seed0"
    run_experiment(cfg, str(out_dir), 0)
    import json
    m = json.load(open(tmp_path / "res" / "exp" / "pu" / "cc_X2Y" / "stats_mlp" / "seed0" / "metrics.json"))
    assert m["trained_in"] == "seen_X" and m["variant"] == "cc_X2Y"
    assert m["data"]["protocol"] == "recording" and m["test"]["clean"]["n"] > 0
    assert (tmp_path / "res" / "exp" / "pu" / "cc_X2Y" / "stats_mlp" / "seed0" / "test_probs.npz").exists()
    assert not (tmp_path / "res" / "exp" / "pu" / "cc_bad").exists()


def test_oom_retry_never_below_min_micro_batch(pu_root, tmp_path, monkeypatch):
    """batch 32 -> retry with micro-batch 16 is allowed; micro-batch 8 is refused (BatchNorm)."""
    import torch
    import phyq.runner as runner
    calls = []

    def fake(cfg, *a, **k):
        calls.append(int(cfg["train"].get("grad_accum", 1)))
        raise torch.cuda.OutOfMemoryError("fake")

    monkeypatch.setattr(runner, "_build_and_train", fake)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    cfg = build_run_config("pu", "stats_mlp")
    cfg["data"] = dict(cfg["data"], **_cfg(pu_root, {"protocol": "recording"}, conditions=[CONDS[0]])["data"])
    cfg["data"].pop("optional_bearings", None)
    cfg.update(device="cpu", _variant="x", results_dir=str(tmp_path / "r"))
    cfg["train"]["batch_size"] = 32
    with pytest.raises(RuntimeError, match="min_micro_batch"):
        runner.run_experiment(cfg, str(tmp_path / "r" / "e" / "pu" / "x" / "m" / "seed0"), 0)
    assert calls == [1, 2]


def test_phyq_paper_runs_in_float32():
    """v9.1: float16 overflows in the quadratic convolutions (activations reach ~4e4 at init; fp16 max 65504)."""
    assert build_run_config("pu", "phyq_paper")["train"]["amp"] is False
    assert build_run_config("cwru", "phyq_paper", model_overrides=["model.quad=false"])["train"]["amp"] is False
    assert build_run_config("pu", "wdcnn")["train"]["amp"] is True


def test_nan_loss_under_amp_restarts_in_float32(pu_root, tmp_path, monkeypatch):
    import phyq.runner as runner
    from phyq.engine.trainer import NonFiniteLoss
    seen = []
    real = runner._build_and_train

    def fake(cfg, *a, **k):
        seen.append(cfg["train"]["amp"])
        if cfg["train"]["amp"]:
            raise NonFiniteLoss("non-finite training loss (AMP float16)", True)
        return real(cfg, *a, **k)

    monkeypatch.setattr(runner, "_build_and_train", fake)
    cfg = build_run_config("pu", "stats_mlp")
    cfg["data"] = dict(cfg["data"], **_cfg(pu_root, {"protocol": "recording"}, conditions=[CONDS[0]])["data"])
    cfg["data"].pop("optional_bearings", None)
    cfg.update(device="cpu", _variant="x", results_dir=str(tmp_path / "r"))
    cfg["train"].update(epochs=1, min_epochs=1, batch_size=8, amp=True)
    cfg["selection"].update(batch_size=16, snrs=["clean"])
    cfg["eval"].update(batch_size=16, snrs=[0], noise_types=["white"])
    out = runner.run_experiment(cfg, str(tmp_path / "r" / "e" / "pu" / "x" / "m" / "seed0"), 0)
    assert seen == [True, False] and out["test"]["clean"]["n"] > 0


def test_trainer_refuses_nan_loss():
    import torch
    from types import SimpleNamespace
    from phyq.engine.trainer import NonFiniteLoss, train

    class NaNNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return torch.full((x.shape[0], 3), float("nan")) + self.w

    class DS(torch.utils.data.Dataset):
        labels = [0, 1, 2, 0]

        def __len__(self):
            return 4

        def __getitem__(self, i):
            return torch.randn(1, 64), self.labels[i]

    data = SimpleNamespace(train=DS(), val=DS(), class_names=["a", "b", "c"], ref_power=torch.ones(1))
    cfg = {"train": {"epochs": 1, "batch_size": 2, "amp": False, "noise_aug": {"enabled": False}}}
    with pytest.raises(NonFiniteLoss):
        train(NaNNet(), data, cfg, torch.device("cpu"), 0)


def test_cuda_error_classification():
    import torch
    from phyq.runner import is_cuda_fault, is_cuda_oom
    assert is_cuda_oom(RuntimeError("CUDA error: out of memory"))
    assert is_cuda_oom(torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert is_cuda_fault(RuntimeError("CUDA error: an illegal memory access was encountered"))
    assert not is_cuda_fault(RuntimeError("CUDA error: out of memory"))
    assert not is_cuda_fault(ValueError("bad"))


def test_plan_micro_batch_uses_measured_memory(monkeypatch):
    import torch
    from types import SimpleNamespace
    import phyq.runner as runner
    GB, MB = 2**30, 2**20
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (int(3.3 * GB), 4 * GB))
    dev = SimpleNamespace(type="cuda")
    # PhyQ-TransNet @8192 fp32: ~80 MB/sample -> batch 32 does not fit a 4 GB laptop GPU -> 16 x 2
    monkeypatch.setattr(runner, "_probe_train_peak", lambda m, C, L, b, amp, d: 80 * MB * b)
    cfg = {"train": {"batch_size": 32, "min_micro_batch": 16, "amp": False}}
    runner.plan_micro_batch(None, cfg, dev, 1, 8192)
    assert cfg["train"]["grad_accum"] == 2
    # @2048: ~16 MB/sample -> batch 64 fits -> no accumulation
    monkeypatch.setattr(runner, "_probe_train_peak", lambda m, C, L, b, amp, d: 16 * MB * b)
    cfg = {"train": {"batch_size": 64, "min_micro_batch": 16, "amp": False}}
    runner.plan_micro_batch(None, cfg, dev, 1, 2048)
    assert cfg["train"]["grad_accum"] == 1
    # estimate above the budget even at micro-batch 16, but the probe with 16 succeeded -> run at 16 x 2
    # (v9.5: before, WDCNN/QCNN were refused because cuDNN benchmark workspaces inflate the estimate)
    monkeypatch.setattr(runner, "_probe_train_peak", lambda m, C, L, b, amp, d: 100 * MB * b)
    cfg = {"train": {"batch_size": 32, "min_micro_batch": 16}}
    runner.plan_micro_batch(None, cfg, dev, 1, 8192)
    assert cfg["train"]["grad_accum"] == 2
    # the probe itself runs out of memory at micro-batch 16 -> clear error
    def oom(*a):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1 GiB")
    monkeypatch.setattr(runner, "_probe_train_peak", oom)
    with pytest.raises(RuntimeError, match="already with micro-batch 16"):
        runner.plan_micro_batch(None, {"train": {"batch_size": 32, "min_micro_batch": 16}}, dev, 1, 8192)
    # CPU: no-op
    cfg = {"train": {"batch_size": 32}}
    runner.plan_micro_batch(None, cfg, torch.device("cpu"), 1, 8192)
    assert "grad_accum" not in cfg["train"]


def test_isolated_run_retries_cuda_fault_with_safe_mode(pu_root, tmp_path):
    """run_suite on the GPU: each run in a fresh process; a CUDA fault is retried with cuda_safe+1."""
    import importlib.util
    import json
    import os
    from types import SimpleNamespace
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("run_suite", os.path.join(here, "scripts", "run_suite.py"))
    rs = importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(0, os.path.join(here, "scripts"))
    spec.loader.exec_module(rs)
    rs.time.sleep = lambda s: None
    cfg = build_run_config("pu", "stats_mlp")
    cfg["data"] = dict(cfg["data"], **_cfg(pu_root, {"protocol": "recording"}, conditions=[CONDS[0]])["data"])
    cfg["data"].pop("optional_bearings", None)
    cfg.update(device="cpu", _variant="x", results_dir=str(tmp_path / "r"), _test_fault_below_safe=1)
    cfg["train"].update(epochs=1, min_epochs=1, batch_size=8)
    cfg["selection"].update(batch_size=16, snrs=["clean"])
    cfg["eval"].update(batch_size=16, snrs=[0], noise_types=["white"])
    out_dir = str(tmp_path / "r" / "e" / "pu" / "x" / "m" / "seed0")
    st = rs.run_isolated(SimpleNamespace(cfg=cfg, out_dir=out_dir, seed=0), force=False)
    assert st["kind"] == "ok"
    with open(os.path.join(out_dir, "metrics.json")) as f:
        assert json.load(f)["cuda_safe"] == 1
    cfg["_test_fault_below_safe"] = 9          # never fixed -> gives up after level 2, reports the fault
    st = rs.run_isolated(SimpleNamespace(cfg=cfg, out_dir=out_dir + "b", seed=0), force=False)
    assert st["kind"] == "cuda_fault"


def test_cudnn_find_failure_is_retryable_and_safe_mode_disables_benchmark():
    import torch
    from phyq.runner import _seed, is_cuda_fault
    assert is_cuda_fault(RuntimeError("FIND was unable to find an engine to execute this computation"))
    _seed({"deterministic": False}, 0)
    assert torch.backends.cudnn.benchmark is True          # default: fastest kernels (benchmark/FIND)
    _seed({"deterministic": False, "cuda_safe": 1}, 0)
    assert torch.backends.cudnn.benchmark is False         # retry: heuristic algorithm choice
    torch.backends.cudnn.benchmark = False


def test_trainer_select_last_runs_all_epochs_without_selection():
    import torch
    from types import SimpleNamespace
    from phyq.engine.trainer import train

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.f = torch.nn.Linear(64, 3)

        def forward(self, x):
            return self.f(x[:, 0])

    class DS(torch.utils.data.Dataset):
        labels = [0, 1, 2, 0, 1, 2, 0, 1]

        def __len__(self):
            return 8

        def __getitem__(self, i):
            return torch.randn(1, 64), self.labels[i]

    data = SimpleNamespace(train=DS(), val=DS(), class_names=["a", "b", "c"], ref_power=torch.ones(1))
    base = {"epochs": 4, "batch_size": 4, "amp": False, "noise_aug": {"enabled": False},
            "patience": 0, "min_epochs": 1, "eval_every": 4}
    cfg = {"train": dict(base, select="last"), "selection": {"snrs": ["clean"], "batch_size": 8}}
    out = train(Net(), data, cfg, torch.device("cpu"), 0)
    assert out["epochs_run"] == 4 and out["best_epoch"] == 4 and out["select"] == "last"
    cfg["train"]["select"] = "bogus"
    with pytest.raises(ValueError):
        train(Net(), data, cfg, torch.device("cpu"), 0)
