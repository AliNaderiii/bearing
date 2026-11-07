"""Pre-flight must catch example/missing paths BEFORE any run starts (no tracebacks per run)."""
import phyq.suite as suite
from phyq.preflight import check_data_paths, report


def _fake(roots):
    def build(ds, model, variant, *a, **k):
        return {"data": {"root": roots[ds], "cache_dir": "/nonexistent_cache"}}
    return build


def test_placeholder_and_missing(monkeypatch, tmp_path):
    ok = tmp_path / "cwru"
    ok.mkdir()
    monkeypatch.setattr(suite, "build_run_config", _fake({
        "cwru": [str(ok)], "pu": "D:/Datasets/PU", "ims": str(tmp_path / "nope")}))
    p = check_data_paths(["cwru", "pu", "ims", "synthetic"])
    assert "cwru" not in p and "synthetic" not in p
    assert "EXAMPLE" in p["pu"][0]
    assert "not found" in p["ims"][0]
    assert "PRE-FLIGHT FAILED" in report(p)


def test_cache_inside_data(monkeypatch, tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setattr(suite, "build_run_config",
                        lambda *a, **k: {"data": {"root": str(d), "cache_dir": str(d / "cache")}})
    assert "INSIDE" in check_data_paths(["cwru"])["cwru"][0]
