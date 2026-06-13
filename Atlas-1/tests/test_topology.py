"""Phase 0 topology pipeline tests."""

import numpy as np
import pytest

from atlas1.synthetic_clouds import arc_cloud, blob_cloud, circle_cloud
from atlas1.topology import (
    CoordinateType,
    ReadoutConfig,
    circular_r2,
    intrinsic_dimension,
    persistence_summary,
    readout_family,
)


@pytest.fixture
def config() -> ReadoutConfig:
    return ReadoutConfig(loop_crit=0.3)


def test_circle_has_persistent_loop(config):
    cloud, _ = circle_cloud(n=400, seed=0)
    b0, b1, loop_score = persistence_summary(cloud, config)
    assert b0 == 1
    assert b1 >= 1
    assert loop_score > config.loop_crit


def test_blobs_have_multiple_components(config):
    cloud, _ = blob_cloud(n_clusters=3, seed=1)
    b0, b1, loop_score = persistence_summary(cloud, config)
    assert 3 <= b0 <= 5
    assert loop_score < config.loop_crit


def test_circle_readout_coordinate(config):
    cloud, theta = circle_cloud(n=400, seed=2)
    readout = readout_family(0, cloud, config, ground_truth=theta)
    assert readout.coordinate_type == CoordinateType.CIRCULAR
    assert readout.coordinate_fidelity is not None
    assert readout.coordinate_fidelity >= 0.8


def test_arc_prefers_non_circular_coordinate(config):
    cloud, t = arc_cloud(n=300, seed=3)
    readout = readout_family(1, cloud, config, ground_truth=t)
    assert readout.loop_score < config.loop_crit
    assert readout.coordinate_type in {CoordinateType.LINEAR, CoordinateType.EMBEDDING_2D}


def test_intrinsic_dimension_circle_near_one():
    cloud, _ = circle_cloud(n=400, seed=4)
    id_est = intrinsic_dimension(cloud)
    assert 0.5 <= id_est <= 2.5


def test_circular_r2_perfect_match():
    theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    score = circular_r2(theta + 0.4, theta)
    assert score > 0.99
