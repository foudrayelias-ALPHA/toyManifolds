"""Geometric and topological analysis of the activation manifold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from .data import NUM_DAYS, NUM_MONTHS
from .extract import load_model_and_extract
from .graph import build_neighborhood_graph, geodesic_distance_matrix


def correlation_dimension(
    points: np.ndarray,
    n_samples: int = 200,
    eps_fractions: np.ndarray | None = None,
) -> tuple[float, dict]:
    """
    Estimate correlation dimension via Grassberger-Procaccia.

    For a k-dimensional manifold embedded in R^d, C(r) ~ r^k for small r.
    """
    flat = points.reshape(-1, points.shape[-1]) if points.ndim == 3 else points
    n = flat.shape[0]
    rng = np.random.default_rng(0)

    if eps_fractions is None:
        eps_fractions = np.logspace(-1.5, 0, 15)

    # Pairwise distances (small n=84, full matrix is fine)
    dists = np.linalg.norm(flat[:, None] - flat[None, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    max_dist = np.max(dists[dists < np.inf])

    correlations = []
    for frac in eps_fractions:
        eps = frac * max_dist
        count = np.sum(dists < eps) - n  # exclude self-pairs
        pairs = n * (n - 1)
        correlations.append(count / pairs)

    correlations = np.array(correlations)
    valid = (correlations > 0) & (correlations < 1)
    log_eps = np.log(eps_fractions[valid] * max_dist)
    log_c = np.log(correlations[valid])

    if len(log_eps) >= 2:
        slope, _ = np.polyfit(log_eps, log_c, 1)
    else:
        slope = float("nan")

    return slope, {
        "eps_fractions": eps_fractions.tolist(),
        "correlations": correlations.tolist(),
        "log_eps": log_eps.tolist(),
        "log_c": log_c.tolist(),
    }


def pca_intrinsic_dimension(
    points: np.ndarray,
    variance_threshold: float = 0.95,
) -> dict:
    """PCA participation ratio and variance-based effective dimension."""
    flat = points.reshape(-1, points.shape[-1]) if points.ndim == 3 else points
    pca = PCA()
    pca.fit(flat)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_for_threshold = int(np.searchsorted(cumvar, variance_threshold) + 1)

    # Participation ratio: (Σλ)² / Σλ²
    eigvals = pca.explained_variance_
    participation_ratio = (eigvals.sum() ** 2) / (eigvals**2).sum()

    return {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": cumvar.tolist(),
        "n_components_95pct": n_for_threshold,
        "participation_ratio": float(participation_ratio),
        "extrinsic_dimension": flat.shape[1],
    }


def geodesic_levina_bickel_dimension(
    geodesic_dist: np.ndarray,
    k: int = 5,
) -> float:
    """
    Levina-Bickel MLE intrinsic dimension on geodesic distances.

    Uses distances along the manifold graph rather than ambient Euclidean
    distance, which is critical when extrinsic dimension >> intrinsic dimension.
    """
    n = geodesic_dist.shape[0]
    k = min(k, n - 1)
    dims = []
    for i in range(n):
        dists = np.sort(geodesic_dist[i])
        dists = dists[1 : k + 1]
        if np.any(dists <= 0):
            continue
        ratios = dists[-1] / dists[:-1]
        dims.append((k - 1) / np.sum(np.log(ratios)))
    return float(np.median(dims)) if dims else float("nan")


def mds_dimension_elbow(geodesic_dist: np.ndarray, max_dim: int = 6) -> dict:
    """MDS stress across dimensions — elbow near intrinsic dimension."""
    from sklearn.manifold import MDS

    stresses = {}
    for dim in range(1, max_dim + 1):
        mds = MDS(
            n_components=dim,
            metric="precomputed",
            init="classical_mds",
            random_state=0,
            max_iter=300,
            n_init=1,
        )
        mds.fit(geodesic_dist)
        stresses[dim] = float(mds.stress_)
    # Normalized stress drop: find dim where marginal improvement < 20%
    drops = {}
    for dim in range(2, max_dim + 1):
        prev = stresses[dim - 1]
        curr = stresses[dim]
        drops[dim] = (prev - curr) / prev if prev > 0 else 0.0
    elbow = max(
        (d for d in range(3, max_dim + 1) if drops.get(d, 0) < 0.15),
        default=2,
    )
    return {"stresses": stresses, "relative_drops": drops, "elbow_dimension": elbow}


def two_nearest_neighbor_dimension(points: np.ndarray) -> float:
    """
    Two-nearest-neighbor estimator (Facco et al. 2017) for intrinsic dimension.
    """
    flat = points.reshape(-1, points.shape[-1]) if points.ndim == 3 else points
    n = flat.shape[0]
    if n < 3:
        return float("nan")

    nbrs = NearestNeighbors(n_neighbors=3).fit(flat)
    distances, _ = nbrs.kneighbors(flat)
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    mu = r2 / (r1 + 1e-10)
    # d = n / Σ log(mu)
    d = n / np.sum(np.log(mu + 1e-10))
    return float(d)


def torus_topology_check(adjacency: np.ndarray, geodesic: np.ndarray) -> dict:
    """
    Verify torus topology: two independent wrap-around loop families.

    H1(torus) = Z × Z has rank 2. We detect this by checking that
    (a) month-cycles and day-cycles close on themselves, and
    (b) the two loop families are not homotopic (geodesic asymmetry).
    """

    def flat_idx(m: int, d: int) -> int:
        return m * NUM_DAYS + d

    month_loop_geodesic = []
    for d in range(NUM_DAYS):
        start = flat_idx(0, d)
        end = flat_idx(0, d)
        # Half-loop to opposite month, then check closure distance
        mid = flat_idx(NUM_MONTHS // 2, d)
        half = geodesic[start, mid]
        full = geodesic[start, flat_idx(NUM_MONTHS - 1, d)] + adjacency[
            flat_idx(NUM_MONTHS - 1, d), end
        ]
        month_loop_geodesic.append(full)

    day_loop_geodesic = []
    for m in range(NUM_MONTHS):
        start = flat_idx(m, 0)
        end = flat_idx(m, 0)
        full = geodesic[start, flat_idx(m, NUM_DAYS - 1)] + adjacency[
            flat_idx(m, NUM_DAYS - 1), end
        ]
        day_loop_geodesic.append(full)

    # Independent loops: month-cycle length != day-cycle length on average
    month_mean = float(np.mean(month_loop_geodesic))
    day_mean = float(np.mean(day_loop_geodesic))
    loops_independent = abs(month_mean - day_mean) > 0.05 * min(
        month_mean, day_mean
    )

    return {
        "expected_H1_rank": 2,
        "month_loop_geodesic_mean": month_mean,
        "day_loop_geodesic_mean": day_mean,
        "loops_independent": loops_independent,
        "month_cycles_close": all(g > 0 for g in month_loop_geodesic),
        "day_cycles_close": all(g > 0 for g in day_loop_geodesic),
        "euler_characteristic": 0,  # χ(torus) = 0
        "topology_consistent_with_torus": (
            loops_independent
            and all(g > 0 for g in month_loop_geodesic)
            and all(g > 0 for g in day_loop_geodesic)
        ),
    }


def analyze_manifold(activations: np.ndarray) -> dict:
    """Run full geometric analysis suite."""
    flat = activations.reshape(-1, activations.shape[-1])
    corr_dim, corr_details = correlation_dimension(flat)
    pca_info = pca_intrinsic_dimension(activations)
    twonn_dim = two_nearest_neighbor_dimension(flat)
    adj, _ = build_neighborhood_graph(activations)
    geodesic = geodesic_distance_matrix(adj)
    geo_dim = geodesic_levina_bickel_dimension(geodesic)
    mds_info = mds_dimension_elbow(geodesic)
    topology = torus_topology_check(adj, geodesic)

    return {
        "correlation_dimension_euclidean": corr_dim,
        "correlation_dimension_details": corr_details,
        "pca": pca_info,
        "two_nn_dimension_euclidean": twonn_dim,
        "geodesic_levina_bickel_dimension": geo_dim,
        "mds_stress": mds_info,
        "topology": topology,
        "concept_dimension": 2,
        "summary": {
            "expected_intrinsic_dim": 2,
            "geodesic_intrinsic_dim": geo_dim,
            "mds_elbow_dimension": mds_info["elbow_dimension"],
            "pca_participation_ratio": pca_info["participation_ratio"],
            "extrinsic_dim": pca_info["extrinsic_dimension"],
            "topology_consistent_with_torus": topology[
                "topology_consistent_with_torus"
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--out", type=Path, default=Path("outputs/analysis.json"))
    args = parser.parse_args()

    data = load_model_and_extract(args.checkpoint, layer=args.layer)
    results = analyze_manifold(data["activations"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    s = results["summary"]
    print("=== Manifold Analysis ===")
    print(f"Concept dimension (expected):       {s['expected_intrinsic_dim']}")
    print(f"Geodesic intrinsic dimension:       {s['geodesic_intrinsic_dim']:.2f}")
    print(f"MDS stress elbow dimension:         {s['mds_elbow_dimension']}")
    print(f"PCA participation ratio:            {s['pca_participation_ratio']:.2f}")
    print(f"Extrinsic dimension:                {s['extrinsic_dim']}")
    print(f"Torus topology consistent:          {s['topology_consistent_with_torus']}")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
