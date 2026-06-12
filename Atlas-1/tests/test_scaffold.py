"""Smoke tests for project scaffolding."""

from atlas1.featurizer import FeaturizerConfig, GroupSparseAutoencoder
from atlas1.metrics import family_count_error, fraction_variance_unexplained
from testbeds.stage0_planted import Stage0Config, generate_stage0


def test_featurizer_forward():
    import torch

    cfg = FeaturizerConfig(ambient_dim=16, num_groups=4, group_size=4)
    model = GroupSparseAutoencoder(cfg)
    x = torch.randn(32, 16)
    x_hat, a = model(x)
    assert x_hat.shape == x.shape
    assert a.shape == (32, 16)


def test_stage0_generates_data():
    x, families = generate_stage0(Stage0Config(ambient_dim=32, num_samples=100, seed=0))
    assert x.shape == (100, 32)
    assert len(families) > 0


def test_m1_torus_ambiguity():
    assert family_count_error(11, 10, torus_ambiguity=True) == 0.0
    assert family_count_error(12, 10, torus_ambiguity=True) == 2.0


def test_fvu_identity():
    import torch

    x = torch.ones(4, 8)
    assert fraction_variance_unexplained(x, x) == 0.0
