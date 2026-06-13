"""Manifold readout: topology, intrinsic dimension, coordinates (§3.2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import ripser
from dreimac import CircularCoords
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors

try:
    import skdim
except ImportError:  # pragma: no cover
    skdim = None


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
    loop_crit: float = 0.3
    max_rips_dim: int = 1
    pca_dims: int = 10
    fps_cap: int = 1_000
    persistence_threshold: float = 0.05
    id_threshold_1d: float = 1.5
    use_dtm_fallback: bool = True


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


def subsample_fps(cloud: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    """Farthest-point subsample for persistence efficiency."""
    n = cloud.shape[0]
    if n <= max_points:
        return cloud

    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(0, n))]
    dists = np.full(n, np.inf)

    for _ in range(max_points - 1):
        last = cloud[idx[-1]]
        dists = np.minimum(dists, np.linalg.norm(cloud - last, axis=1))
        idx.append(int(np.argmax(dists)))

    return cloud[np.array(idx)]


def pca_reduce(cloud: np.ndarray, n_components: int) -> np.ndarray:
    """Reduce ambient dimension before Rips / ID estimation."""
    n_components = min(n_components, cloud.shape[0] - 1, cloud.shape[1])
    if n_components < 1:
        return cloud
    return PCA(n_components=n_components, random_state=0).fit_transform(cloud)


def _estimate_b0_components(reduced: np.ndarray, percentile: float = 20.0) -> int:
    """Connected components at a local neighborhood scale."""
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial.distance import pdist, squareform

    distances = squareform(pdist(reduced))
    positive = distances[distances > 0]
    if positive.size == 0:
        return 1
    epsilon = float(np.percentile(positive, percentile))
    adjacency = (distances <= epsilon).astype(np.int8)
    n_components, _ = connected_components(adjacency, directed=False)
    return int(n_components)


def _count_persistent_features(dgm: np.ndarray, threshold: float) -> int:
    finite = dgm[np.isfinite(dgm).all(axis=1)]
    if finite.size == 0:
        return 0
    persistences = finite[:, 1] - finite[:, 0]
    return int(np.sum(persistences > threshold))


def _ripser_summary(reduced: np.ndarray, config: ReadoutConfig) -> tuple[int, int, float]:
    dgms = ripser.ripser(reduced, maxdim=config.max_rips_dim)["dgms"]
    h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))

    b0 = _estimate_b0_components(reduced)
    b1 = _count_persistent_features(h1, config.persistence_threshold)

    if h1.size:
        loop_score = float(np.max(h1[:, 1] - h1[:, 0]))
    else:
        loop_score = 0.0

    return b0, b1, loop_score


def _dtm_summary(cloud: np.ndarray, config: ReadoutConfig) -> tuple[int, int, float]:
    from gtda.homology import WeightedRipsPersistence

    wrp = WeightedRipsPersistence(
        homology_dimensions=(0, 1),
        weights="DTM",
        weight_params={"n_neighbors": min(30, cloud.shape[0] - 1)},
    )
    diagrams = wrp.fit_transform(cloud[None, :, :])
    h1 = diagrams[0, diagrams[0][:, 2] == 1][:, :2]

    b0 = _estimate_b0_components(cloud)
    b1 = _count_persistent_features(h1, config.persistence_threshold)
    loop_score = float(np.max(h1[:, 1] - h1[:, 0])) if len(h1) else 0.0
    return b0, b1, loop_score


def persistence_summary(cloud: np.ndarray, config: ReadoutConfig) -> tuple[int, int, float]:
    """Vietoris–Rips persistence with optional DTM fallback."""
    reduced = pca_reduce(subsample_fps(cloud, config.fps_cap), config.pca_dims)
    try:
        return _ripser_summary(reduced, config)
    except Exception:
        if not config.use_dtm_fallback:
            raise
        return _dtm_summary(reduced, config)


def _participation_ratio(cloud: np.ndarray) -> float:
    ev = PCA().fit(cloud).explained_variance_
    return float((ev.sum() ** 2) / np.square(ev).sum())


def _twonn_manual(reduced: np.ndarray) -> float:
    n = reduced.shape[0]
    if n < 4:
        return float("nan")
    nbrs = NearestNeighbors(n_neighbors=3).fit(reduced)
    distances, _ = nbrs.kneighbors(reduced)
    mu = distances[:, 2] / (distances[:, 1] + 1e-10)
    return float(n / np.sum(np.log(mu + 1e-10)))


def intrinsic_dimension(cloud: np.ndarray, *, pca_dims: int = 10) -> float:
    """TwoNN intrinsic dimension on PCA-reduced cloud (§3.2 step 2)."""
    reduced = pca_reduce(cloud, pca_dims)
    estimate = _twonn_manual(reduced)

    if skdim is not None:
        try:
            sk_estimate = float(skdim.id.TwoNN().fit(reduced).dimension_)
            if np.isfinite(sk_estimate):
                estimate = sk_estimate
        except Exception:
            pass

    if not np.isfinite(estimate) or estimate > cloud.shape[1]:
        estimate = max(1.0, _participation_ratio(reduced) / 2.0)

    return float(np.clip(estimate, 0.5, cloud.shape[1]))


def circular_r2(recovered: np.ndarray, ground_truth: np.ndarray) -> float:
    """Circular R² via sin/cos regression against a ground-truth angle."""
    gt = np.asarray(ground_truth).reshape(-1)
    rec = np.asarray(recovered).reshape(-1)
    if len(gt) != len(rec):
        raise ValueError("coordinate lengths must match")

    from sklearn.linear_model import LinearRegression

    design = np.column_stack([np.sin(rec), np.cos(rec)])
    sin_score = LinearRegression().fit(design, np.sin(gt)).score(design, np.sin(gt))
    cos_score = LinearRegression().fit(design, np.cos(gt)).score(design, np.cos(gt))
    return float((sin_score + cos_score) / 2.0)


def _fit_circular_coordinate(cloud: np.ndarray) -> np.ndarray:
    n = cloud.shape[0]
    n_landmarks = int(np.clip(n // 8, 30, min(100, n - 1)))
    cc = CircularCoords(cloud, n_landmarks=n_landmarks)
    best_coords = cc.get_coordinates(perc=0.5)

    if n >= 120:
        best_score = -np.inf
        for perc in (0.3, 0.5, 0.7):
            coords = cc.get_coordinates(perc=perc)
            design = np.column_stack([np.sin(coords), np.cos(coords)])
            score = np.linalg.svd(design, compute_uv=False)[0]
            if score > best_score:
                best_score = score
                best_coords = coords

    return np.mod(best_coords, 2 * np.pi)


def _fit_linear_coordinate(cloud: np.ndarray) -> np.ndarray:
    reduced = pca_reduce(cloud, 1).reshape(-1)
    if np.ptp(reduced) < 1e-8:
        return reduced
    return (reduced - reduced.min()) / (reduced.max() - reduced.min())


def _fit_embedding_2d(cloud: np.ndarray) -> np.ndarray:
    if cloud.shape[0] >= 20:
        try:
            return Isomap(n_components=2, n_neighbors=min(15, cloud.shape[0] - 1)).fit_transform(cloud)
        except Exception:
            pass
    return pca_reduce(cloud, 2)


def fit_coordinate(
    cloud: np.ndarray,
    loop_score: float,
    intrinsic_dim: float,
    config: ReadoutConfig,
) -> tuple[CoordinateType, np.ndarray]:
    """Circular (dreimac), 1-D PCA/Isomap, or 2-D embedding (§3.2 step 3)."""
    if loop_score > config.loop_crit:
        return CoordinateType.CIRCULAR, _fit_circular_coordinate(cloud)

    if intrinsic_dim <= config.id_threshold_1d:
        return CoordinateType.LINEAR, _fit_linear_coordinate(cloud)

    return CoordinateType.EMBEDDING_2D, _fit_embedding_2d(cloud)


def readout_family(
    group_id: int,
    cloud: np.ndarray,
    config: ReadoutConfig | None = None,
    *,
    ground_truth: Optional[np.ndarray] = None,
) -> TopologyReadout:
    """Full per-family readout pipeline."""
    config = config or ReadoutConfig()
    b0, b1, loop_score = persistence_summary(cloud, config)
    id_est = intrinsic_dimension(cloud, pca_dims=config.pca_dims)
    coord_type, coord = fit_coordinate(cloud, loop_score, id_est, config)

    fidelity = None
    if ground_truth is not None and coord is not None:
        if coord_type == CoordinateType.CIRCULAR:
            fidelity = circular_r2(coord, ground_truth)
        else:
            from atlas1.metrics import coordinate_fidelity

            fidelity = coordinate_fidelity(coord, ground_truth)

    return TopologyReadout(
        group_id=group_id,
        b0=b0,
        b1=b1,
        intrinsic_dim=id_est,
        usage=float(len(cloud)),
        loop_score=loop_score,
        coordinate_type=coord_type,
        coordinate=coord,
        coordinate_fidelity=fidelity,
    )
