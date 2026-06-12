"""Manifold readout: topology, intrinsic dimension, coordinates (§3.2).

Shared telemetry surface with LOOP-1. Implementations are stubs until Phase 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

import numpy as np


class CoordinateType(str, Enum):
    CIRCULAR = "circular"
    LINEAR = "linear"
    EMBEDDING_2D = "embedding_2d"
    NONE = "none"


@dataclass
class TopologyReadout:
    """Per-family manifold summary reported in the atlas (§3.3)."""

    group_id: int
    b0: int
    b1: int
    intrinsic_dim: float
    usage: float
    loop_score: float
    coordinate_type: CoordinateType
    coordinate: Optional[np.ndarray] = None
    coordinate_fidelity: Optional[float] = None


@dataclass
class ReadoutConfig:
    theta_act: float = 0.1
    loop_crit: float = 0.5
    max_rips_dim: int = 1
    pca_dims: int = 10
    fps_cap: int = 1_000


def family_cloud(
    decoder_weight: np.ndarray,
    activations: np.ndarray,
    group_slice: slice,
) -> np.ndarray:
    """Normalized group reconstructions for active inputs (§3.2 step 1)."""
    group_atoms = decoder_weight[:, group_slice]
    group_acts = activations[:, group_slice]
    recon = group_acts @ group_atoms.T
    norms = np.linalg.norm(recon, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return recon / norms


def persistence_summary(cloud: np.ndarray, config: ReadoutConfig) -> tuple[int, int, float]:
    """Vietoris–Rips persistence → (b0, b1, loop_score). Stub until ripser wiring."""
    _ = (cloud, config)
    raise NotImplementedError("Phase 0: wire ripser / DTM fallback")


def intrinsic_dimension(cloud: np.ndarray) -> float:
    """TwoNN intrinsic dimension estimate. Stub until scikit-dimension wiring."""
    _ = cloud
    raise NotImplementedError("Phase 0: wire scikit-dimension TwoNN")


def fit_coordinate(
    cloud: np.ndarray,
    loop_score: float,
    intrinsic_dim: float,
    config: ReadoutConfig,
) -> tuple[CoordinateType, np.ndarray]:
    """Circular (dreimac), 1-D PCA/Isomap, or 2-D embedding (§3.2 step 3)."""
    _ = (cloud, loop_score, intrinsic_dim, config)
    raise NotImplementedError("Phase 0: coordinate extraction pipeline")


def readout_family(
    group_id: int,
    cloud: np.ndarray,
    config: ReadoutConfig,
) -> TopologyReadout:
    """Full per-family readout pipeline."""
    b0, b1, loop_score = persistence_summary(cloud, config)
    id_est = intrinsic_dimension(cloud)
    coord_type, coord = fit_coordinate(cloud, loop_score, id_est, config)
    return TopologyReadout(
        group_id=group_id,
        b0=b0,
        b1=b1,
        intrinsic_dim=id_est,
        usage=float(len(cloud)),
        loop_score=loop_score,
        coordinate_type=coord_type,
        coordinate=coord,
    )
