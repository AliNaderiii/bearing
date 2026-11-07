import pytest
import torch

from phyq.models import build_model
from phyq.models.layers import QuadConv1d

BASE = {"name": "phyqnet", "width": 0.25}


@pytest.mark.parametrize("name", ["wdcnn", "qcnn", "drsn", "cnn_transformer", "stats_mlp"])
def test_baselines_forward(name):
    m = build_model({"name": name}, 2, 5, 2048).eval()
    assert m(torch.randn(3, 2, 2048)).shape == (3, 5)


@pytest.mark.parametrize("ov", [
    {}, {"conv": "linear"}, {"conv": "quad_gated"}, {"quad_scope": "stem"}, {"attention": "se_temporal"},
    {"attention": "none", "act": "relu"}, {"shrink": True}, {"transformer": {"enabled": False}},
    {"stats": {"enabled": False}}, {"envelope": {"enabled": True}}, {"med": {"enabled": True}},
    {"fusion": "concat", "envelope": {"enabled": True}}, {"multiscale": None},
    {"transformer": {"enabled": True, "pos": "sin"}},
])
def test_phyqnet_variants(ov):
    cfg = dict(BASE, **ov)
    m = build_model(cfg, 1, 10, 2048).eval()
    y = m(torch.randn(2, 1, 2048))
    assert y.shape == (2, 10) and torch.isfinite(y).all()


def test_quadconv_starts_linear():
    q = QuadConv1d(2, 4, 5, padding=2)
    x = torch.randn(3, 2, 64)
    assert torch.allclose(q(x), q.r(x), atol=1e-6)


def test_backward():
    m = build_model(dict(BASE, envelope={"enabled": True}), 1, 4, 2048)
    loss = m(torch.randn(4, 1, 2048)).logsumexp(1).mean()
    loss.backward()
    assert all(p.grad is not None for n, p in m.named_parameters() if p.requires_grad and "pos" not in n)
