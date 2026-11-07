import torch

from phyq.features.stats import n_features, stat_features


def test_shape():
    assert stat_features(torch.randn(4, 3, 2048)).shape == (4, 3 * n_features(True))
    assert stat_features(torch.randn(4, 3, 2048), amplitude=False).shape == (4, 3 * n_features(False))


def test_invariant_features_are_scale_invariant():
    x = torch.randn(4, 1, 2048) + 0.3 * torch.sin(torch.linspace(0, 200, 2048))
    a = stat_features(x, amplitude=False)
    b = stat_features(7.5 * x, amplitude=False)
    assert torch.allclose(a, b, atol=1e-3)


def test_kurtosis_gaussian():
    x = torch.randn(64, 1, 8192)
    f = stat_features(x, amplitude=False).view(64, -1)
    assert abs(f[:, 1].exp().mean().item() - 3.0) < 0.1
