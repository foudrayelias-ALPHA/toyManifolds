"""Shape/behaviour tests for cross_model_alignment.

Run: PYTHONPATH=<session>/code uv run pytest <session>/code/methods/cross_model_alignment/tests -v
"""

from __future__ import annotations

import numpy as np
import pytest

from methods.cross_model_alignment import (
    coupling_to_matching,
    gromov_wasserstein,
    label_recovery,
    null_distribution,
    procrustes_align,
)

_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _ring(n: int, seed: int = 0, jitter: float = 0.04) -> np.ndarray:
    """Points on a (slightly irregular) ring — jitter breaks the perfect dihedral symmetry."""
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    return X + rng.normal(scale=jitter, size=X.shape)


def _dist(X: np.ndarray) -> np.ndarray:
    return np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)


def test_gw_identity_recovers_permutation():
    D = _dist(_ring(7, seed=1))
    out = gromov_wasserstein(D, D, epsilon=5e-3, n_init=3, seed=0)
    assert out["coupling"].shape == (7, 7)
    matching = coupling_to_matching(out["coupling"])
    assert sorted(matching.tolist()) == list(range(7))  # a permutation
    rec = label_recovery(matching, _LABELS, _LABELS, symmetry_group="cyclic")
    assert rec["mod_symmetry"] == pytest.approx(1.0)


def test_label_recovery_rotation_is_symmetry():
    # target relabelled by a +2 rotation: source i -> (i+2) mod 7
    matching = np.array([(i + 2) % 7 for i in range(7)])
    rec = label_recovery(matching, _LABELS, _LABELS, symmetry_group="cyclic")
    assert rec["raw"] < 1.0                       # absolute phase wrong
    assert rec["mod_symmetry"] == pytest.approx(1.0)  # but a perfect rotation
    assert rec["best_group_element"][0] == "rotate"


def test_label_recovery_none_group():
    matching = np.array([(i + 2) % 7 for i in range(7)])
    rec = label_recovery(matching, _LABELS, _LABELS, symmetry_group="none")
    assert rec["mod_symmetry"] == pytest.approx(rec["raw"])


def test_procrustes_identity_zero_residual():
    X = _ring(7, seed=2)
    out = procrustes_align(X, X, np.arange(7))
    assert out["R"].shape == (2, 2)
    assert out["residual"] < 1e-6


def test_null_mean_exceeds_true_gw():
    D = _dist(_ring(7, seed=3))
    true_gw = gromov_wasserstein(D, D, epsilon=5e-3, n_init=3, seed=0)["gw_distance"]
    nd = null_distribution(D, D, kind="shuffle", n_samples=12, seed=0, gw_epsilon=5e-3, gw_n_init=2)
    assert nd["mean"] > true_gw
