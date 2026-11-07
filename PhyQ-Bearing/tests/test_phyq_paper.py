"""PhyQ-TransNet (the author's model): architecture identity, ablation switches, Neural-MED fitting."""
import numpy as np
import pytest
import torch

from phyq.models import build_model, count_params
from phyq.suite import CFG_DIR
from phyq.utils.config import load_config
import os


def _cfg(**ov):
    cfg = load_config(os.path.join(CFG_DIR, "models", "phyq_paper.yaml"))["model"]
    cfg.update(ov)
    return cfg


def test_parameter_count_identical_to_paper_model():
    # original QDCA_Hybrid_Pro(in_channels=3, num_classes=10, feat_dim=33) has 1,404,715 parameters
    m = build_model(_cfg(), 3, 10, 2048)
    assert count_params(m) == 1_404_715


def test_state_dict_keys_match_original_layout():
    keys = set(build_model(_cfg(), 3, 10, 2048).state_dict())
    for k in ["stem.0.conv_r.weight", "stem.0.conv_g.weight", "stem.0.conv_b.weight", "stem.0.c",
              "branch_1.0.conv_r.weight", "deep_stage.0.attn.conv_w.weight", "deep_stage.2.shrinkage.fc.0.weight",
              "pos_embedding", "transformer.layers.1.linear1.weight", "aux_net.0.weight", "fusion_gate.0.weight",
              "classifier.2.weight"]:
        assert k in keys, k


@pytest.mark.parametrize("ov", [{}, {"quad": False}, {"attention": False}, {"transformer": False},
                                {"stats": False}, {"shrink": False}, {"act": "relu"},
                                {"prefilter": {"enabled": False}}, {"input_norm": "none"}])
def test_ablation_variants_forward_backward(ov):
    m = build_model(_cfg(**ov), 2, 4, 2048)
    y = m(torch.randn(4, 2, 2048))
    assert y.shape == (4, 4) and torch.isfinite(y).all()
    y.logsumexp(1).mean().backward()


def test_long_window_pu():
    m = build_model(_cfg(), 1, 3, 8192).eval()
    assert m(torch.randn(2, 1, 8192)).shape == (2, 3)


def test_prefilter_fitted_on_train_then_frozen():
    m = build_model(_cfg(prefilter={"enabled": True, "epochs": 1, "batch_size": 16}), 2, 3, 1024)
    n_trainable = count_params(m)
    x = np.random.RandomState(0).randn(40, 2, 1024).astype(np.float32)
    x[:, :, ::100] += 8.0                                        # impulses -> kurtosis gradient
    before = m.prefilter[0].gate.weight.clone()
    m.fit_prefilter(x, seed=0)
    assert bool(m.prefilter_fitted)
    assert not torch.equal(before, m.prefilter[0].gate.weight)  # it was trained
    assert all(not p.requires_grad for p in m.prefilter.parameters())  # and frozen afterwards
    assert count_params(m) == n_trainable                       # not part of the trainable model
    assert m(torch.randn(3, 2, 1024)).shape == (3, 3)
