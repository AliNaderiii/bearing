import numpy as np
import torch

from phyq.data.noise import add_noise, eval_generator


def _snr(x, y):
    n = y - x
    return 10 * torch.log10(x.pow(2).mean(-1) / n.pow(2).mean(-1))


def test_relative_snr_exact():
    x = torch.randn(16, 2, 2048) * torch.rand(16, 2, 1) * 5
    for kind in ("white", "pink", "impulsive"):
        for snr in (-10, 0, 7):
            y = add_noise(x, snr, kind, "relative", gen=eval_generator(0, snr, kind, 0))
            assert torch.allclose(_snr(x, y), torch.full((16, 2), float(snr)), atol=1e-3), kind


def test_absolute_snr_uses_reference_power():
    x = torch.randn(64, 1, 2048) * 0.01           # weak signal
    ref = torch.tensor([1.0])                       # train clean power
    y = add_noise(x, 0.0, "white", "absolute", ref, gen=eval_generator(0, 0, "white", 0))
    pn = (y - x).pow(2).mean().item()
    assert abs(pn - 1.0) < 1e-3                      # noise power set by reference, not by window


def test_eval_noise_deterministic_and_paired():
    x = torch.randn(8, 1, 1024)
    a = add_noise(x, -6, "white", "relative", gen=eval_generator(1, -6, "white", 3))
    b = add_noise(x, -6, "white", "relative", gen=eval_generator(1, -6, "white", 3))
    c = add_noise(x, -6, "white", "relative", gen=eval_generator(1, -6, "white", 4))
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_pink_has_1_over_f_spectrum():
    x = torch.zeros(32, 1, 8192)
    y = add_noise(x + 1e-3, 0, "pink", "absolute", torch.tensor([1.0]), gen=eval_generator(0, 0, "pink", 0))
    P = torch.fft.rfft(y[:, 0], dim=-1).abs().pow(2).mean(0).numpy()
    f = np.arange(len(P))
    sel = (f > 10) & (f < 3000)
    slope = np.polyfit(np.log(f[sel]), np.log(P[sel]), 1)[0]
    assert -1.3 < slope < -0.7


def test_train_aug_mixed_mode_runs():
    from phyq.data.noise import TrainNoiseAugment
    aug = TrainNoiseAugment({"enabled": True, "mode": "mixed", "prob": 1.0}, torch.tensor([1.0]))
    y = aug(torch.randn(8, 1, 1024))
    assert y.shape == (8, 1, 1024) and torch.isfinite(y).all()


def test_bootstrap_ci_contains_point_estimate():
    from phyq.analysis.stats_tests import grouped_bootstrap_ci
    rng = np.random.RandomState(0)
    c = (rng.rand(200) < 0.7).astype(float)
    g = np.repeat(np.arange(10), 20)
    lab = np.repeat(np.arange(10), 20)       # one recording per class (CWRU-like)
    r = grouped_bootstrap_ci(c, g, lab)
    assert r["lo"] <= r["acc"] <= r["hi"] and r["hi"] - r["lo"] < 30
