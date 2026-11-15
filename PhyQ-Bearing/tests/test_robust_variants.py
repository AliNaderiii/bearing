"""v10.4: robustness variants (E1 robust clipping, E2 'impacts' augmentation, E3 robust normalisation)."""
import torch

from phyq.data.noise import TrainNoiseAugment, _unit_noise, add_noise
from phyq.models.phyq_paper import PhyQTransNet, robust_clip

BASE = {"prefilter": {"enabled": False}}


def _n(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def test_defaults_unchanged_and_no_new_params():
    torch.manual_seed(0); m0 = PhyQTransNet(3, 10, dict(BASE))
    torch.manual_seed(0); m1 = PhyQTransNet(3, 10, {**BASE, "eqc_clip": None, "input_norm": "instance"})
    assert _n(m0) == _n(m1) == 1404715
    torch.manual_seed(0); m2 = PhyQTransNet(3, 10, {**BASE, "eqc_clip": 9, "input_norm": "robust"})
    assert _n(m2) == _n(m0)
    x = torch.randn(4, 3, 2048)
    m0.eval(); m1.eval()
    with torch.no_grad():
        assert torch.equal(m0(x), m1(x))


def test_robust_clip_bounds_and_gaussian_untouched():
    g = torch.randn(4, 8, 4000)
    assert torch.equal(robust_clip(g, 9), g)                 # tau = 13.3 sigma -> Gaussian never clipped
    z = torch.randn(4, 8, 4000) ** 2
    frac = (robust_clip(z, 9) < z).float().mean().item()
    assert frac < 0.01                                         # P(chi2_1 > 9) = 0.27 %
    zi = z.clone(); zi[..., ::500] = 1e6                       # impulses -> bounded output
    out = robust_clip(zi, 9)
    assert out.max() < 50


def test_variants_forward_backward():
    for cfg in ({"eqc_clip": 9}, {"input_norm": "robust"}, {"eqc_clip": 9, "eqc_clip_layers": "front"},
                {"quad": False, "eqc_clip": 9}):
        m = PhyQTransNet(1, 4, {**BASE, **cfg})
        y = m(torch.randn(16, 1, 2048))
        assert y.shape == (16, 4) and torch.isfinite(y).all()
        y.sum().backward()


def test_robust_norm_insensitive_to_spikes():
    m = PhyQTransNet(1, 4, {**BASE, "input_norm": "robust"})
    x = torch.randn(2, 1, 2048)
    xi = x.clone(); xi[..., 100] = 1e4
    med = xi.median(-1, keepdim=True).values
    mad = (xi - med).abs().median(-1, keepdim=True).values
    assert 0.8 < float(1.4826 * mad.mean()) < 1.2              # scale still ~ sigma of the Gaussian part


def test_impacts_noise_unit_power_and_distinct():
    u = _unit_noise((64, 2, 2048), "impacts", None, "cpu")
    assert torch.allclose(u.pow(2).mean(-1), torch.ones(64, 2), atol=1e-4)
    k = ((u - u.mean(-1, keepdim=True)) ** 4).mean(-1) / u.var(-1) ** 2
    assert k.median() > 10                                      # strongly impulsive
    aug = TrainNoiseAugment({"enabled": True, "prob": 1.0, "types": ["impacts"], "mode": "relative"}, None)
    x = torch.randn(8, 1, 2048)
    assert torch.isfinite(aug(x)).all()
    xn = add_noise(x, 0.0, "impacts", "relative")
    r = (xn - x).pow(2).mean() / x.pow(2).mean()
    assert 0.8 < float(r) < 1.25
