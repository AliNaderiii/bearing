"""v10.5 additions: harmonic / recorded test noise, cache channel guard, fusion ablation."""
import numpy as np
import pytest
import torch

from phyq.data.noise import add_noise, eval_generator, set_recorded_bank, _unit_noise
from phyq.data.recording import Recording, SignalCache
from phyq.models import build_model


def test_harmonic_noise_power_and_determinism():
    x = torch.zeros(4, 2, 2048)
    y = add_noise(x, -6.0, "harmonic", "absolute", ref_power=torch.ones(2), gen=eval_generator(1, -6.0, "harmonic", 0))
    assert abs(float(y.pow(2).mean()) - 10 ** 0.6) < 0.05 * 10 ** 0.6
    n1 = _unit_noise((4, 2, 2048), "harmonic", eval_generator(1, -6.0, "harmonic", 0), "cpu")
    n2 = _unit_noise((4, 2, 2048), "harmonic", eval_generator(1, -6.0, "harmonic", 0), "cpu")
    assert torch.equal(n1, n2)
    assert torch.allclose(n1.pow(2).mean(-1), torch.ones(4, 2), atol=1e-4)
    # periodic: a strong spectral line (peak >> median of the spectrum)
    S = torch.fft.rfft(n1[0, 0]).abs()
    assert S.max() > 20 * S.median()


def test_recorded_noise_uses_bank_and_is_deterministic():
    set_recorded_bank(None)
    with pytest.raises(RuntimeError):
        _unit_noise((2, 1, 256), "recorded", eval_generator(1, 0.0, "recorded", 0), "cpu")
    bank = np.sin(np.arange(20000) * 0.3).astype(np.float32)
    set_recorded_bank(bank)
    n1 = _unit_noise((2, 1, 256), "recorded", eval_generator(1, 0.0, "recorded", 0), "cpu")
    n2 = _unit_noise((2, 1, 256), "recorded", eval_generator(1, 0.0, "recorded", 0), "cpu")
    assert torch.equal(n1, n2)
    assert torch.allclose(n1.pow(2).mean(-1), torch.ones(2, 1), atol=1e-4)
    x = torch.randn(2, 1, 256)
    y = add_noise(x, 0.0, "recorded", "relative", gen=eval_generator(1, 0.0, "recorded", 0))
    assert y.shape == x.shape and torch.isfinite(y).all()
    set_recorded_bank(None)


def test_cache_refuses_channel_mismatch(tmp_path):
    def two():
        return np.zeros((2, 4096), np.float32)

    cache = SignalCache(str(tmp_path))
    r2 = Recording(rid="r", dataset="d", label=0, label_name="a", group="g", condition="c", fs=1000.0,
                   loader=two, channels=["DE", "FE"])
    assert cache.get(r2).shape == (2, 4096)
    r1 = Recording(rid="r", dataset="d", label=0, label_name="a", group="g", condition="c", fs=1000.0,
                   loader=two, channels=["DE"])
    with pytest.raises(RuntimeError, match="channel"):
        SignalCache(str(tmp_path)).shape(r1)


@pytest.mark.parametrize("fusion", ["gated", "mean"])
def test_fusion_option(fusion):
    cfg = {"name": "phyq_paper", "prefilter": {"enabled": False}, "fusion": fusion}
    net = build_model(cfg, 1, 4, 1024).eval()
    with torch.no_grad():
        out = net(torch.randn(3, 1, 1024))
    assert out.shape == (3, 4)
    a = net.last_alpha
    if fusion == "mean":
        assert torch.allclose(a, torch.full_like(a, 0.5))


def test_build_bank_from_mock_nrel(tmp_path):
    from mock_data import make_nrel
    from phyq.data.noise_bank import build_bank
    root = tmp_path / "nrel"
    make_nrel(str(root))
    cfg = {"paths": {"nrel": str(root)}, "data": {"name": "cwru", "cache_dir": str(tmp_path / "cache")},
           "eval": {"recorded_noise": {"dataset": "nrel", "label": "Healthy", "channel": 0}}}
    bank = build_bank(cfg, 12000.0)
    assert bank.ndim == 1 and len(bank) > 4096 and np.isfinite(bank).all()
    assert abs(float(bank.mean())) < 1e-3
    with pytest.raises(ValueError):
        build_bank(dict(cfg, data={"name": "nrel", "cache_dir": str(tmp_path / "cache")}), 12000.0)
