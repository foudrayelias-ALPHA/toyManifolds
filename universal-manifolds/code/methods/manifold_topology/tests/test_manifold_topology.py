"""Topology tests for manifold_topology.

Run: PYTHONPATH=<session>/code uv run pytest <session>/code/methods/manifold_topology/tests -v
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ripser")  # skip gracefully if the PH backend is unavailable

from methods.manifold_topology import betti_match, persistent_homology, significant_betti


def _circle(n: int = 60, seed: int = 0, jitter: float = 0.02) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    return X + rng.normal(scale=jitter, size=X.shape)


def _segment(n: int = 60, seed: int = 0, jitter: float = 0.02) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    X = np.stack([t, np.zeros_like(t)], axis=1)
    return X + rng.normal(scale=jitter, size=X.shape)


def test_circle_has_one_loop():
    out = persistent_homology(_circle(), maxdim=1, rel_threshold=0.25)
    assert out["betti"][0] == 1
    assert out["betti"][1] == 1


def test_segment_has_no_loop():
    out = persistent_homology(_segment(), maxdim=1, rel_threshold=0.25)
    assert out["betti"][0] == 1
    assert out["betti"][1] == 0


def test_significant_betti_filters_noise():
    # one long bar + two tiny noise bars; ref_scale=1.0 → only the long bar is significant
    diagram = np.array([[0.0, 1.0], [0.2, 0.22], [0.3, 0.33]])
    assert significant_betti(diagram, rel_threshold=0.25, ref_scale=1.0) == 1
    # against a large external scale, even the long bar is noise → 0
    assert significant_betti(diagram, rel_threshold=0.25, ref_scale=20.0) == 0


def test_betti_match():
    assert betti_match({0: 1, 1: 1}, {0: 1, 1: 1})["match"] is True
    assert betti_match({0: 1, 1: 1}, {0: 1, 1: 0})["match"] is False
