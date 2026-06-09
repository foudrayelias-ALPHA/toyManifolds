"""Neighborhood graph on the month×day concept grid with cyclic wrap-around."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from .data import NUM_DAYS, NUM_MONTHS


def grid_neighbors(month: int, day: int) -> list[tuple[int, int]]:
    """Four conceptual neighbors with torus wrap-around."""
    return [
        ((month + 1) % NUM_MONTHS, day),
        ((month - 1) % NUM_MONTHS, day),
        (month, (day + 1) % NUM_DAYS),
        (month, (day - 1) % NUM_DAYS),
    ]


def build_neighborhood_graph(
    activations: np.ndarray,
    use_geodesic_weights: bool = True,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Build adjacency matrix for the manifold graph.

    Args:
        activations: (12, 7, d) activation tensor
        use_geodesic_weights: edge weight = Euclidean distance in activation space

    Returns:
        adjacency: (84, 84) symmetric weight matrix
        edges: list of (i, j) index pairs in flattened grid order
    """
    flat = activations.reshape(NUM_MONTHS * NUM_DAYS, -1)
    n = flat.shape[0]
    adj = np.zeros((n, n), dtype=np.float64)
    edges: list[tuple[int, int]] = []

    def flat_idx(m: int, d: int) -> int:
        return m * NUM_DAYS + d

    for m in range(NUM_MONTHS):
        for d in range(NUM_DAYS):
            i = flat_idx(m, d)
            for nm, nd in grid_neighbors(m, d):
                j = flat_idx(nm, nd)
                if i < j:
                    if use_geodesic_weights:
                        w = float(np.linalg.norm(flat[i] - flat[j])) + 1e-6
                    else:
                        w = 1.0
                    adj[i, j] = w
                    adj[j, i] = w
                    edges.append((i, j))

    return adj, edges


def geodesic_distance_matrix(adjacency: np.ndarray) -> np.ndarray:
    """All-pairs shortest path distances along the neighborhood graph."""
    graph = csr_matrix(adjacency)
    dist = shortest_path(graph, directed=False, unweighted=False)
    dist[np.isinf(dist)] = 0.0
    return dist


def separable_baseline_activations(
    activations: np.ndarray,
) -> np.ndarray:
    """
    Optimal separable approximation: a_ij = f_i · v1 + g_j · v2.

    Fits scalar month/day coefficients and two shared directions by least
    squares. The resulting mesh is a flat parallelogram — the linear hypothesis
    baseline for contrast.
    """
    flat = activations.reshape(NUM_MONTHS * NUM_DAYS, -1)
    d = flat.shape[1]

    # Initialize directions from marginal variation
    month_profiles = activations - activations.mean(axis=(0, 1))
    month_profiles = month_profiles.mean(axis=1)  # (12, d)
    day_profiles = (activations - activations.mean(axis=(0, 1))).mean(axis=0)

    _, _, vt = np.linalg.svd(
        np.vstack([month_profiles, day_profiles]), full_matrices=False
    )
    v1 = vt[0]
    v2 = vt[1] if vt.shape[0] > 1 else vt[0] * 0.5

    # Solve for scalar coefficients per grid point
    design = np.zeros((NUM_MONTHS * NUM_DAYS, 2))
    for m in range(NUM_MONTHS):
        for day in range(NUM_DAYS):
            idx = m * NUM_DAYS + day
            design[idx] = [m / NUM_MONTHS, day / NUM_DAYS]

    # Per-dimension separable fit, then project onto v1, v2 subspace
    f = np.zeros(NUM_MONTHS)
    g = np.zeros(NUM_DAYS)
    for m in range(NUM_MONTHS):
        f[m] = np.dot(month_profiles[m], v1)
    for day in range(NUM_DAYS):
        g[day] = np.dot(day_profiles[day], v2)

    separable = np.zeros_like(flat)
    for m in range(NUM_MONTHS):
        for day in range(NUM_DAYS):
            idx = m * NUM_DAYS + day
            separable[idx] = f[m] * v1 + g[day] * v2

    return separable.reshape(NUM_MONTHS, NUM_DAYS, d)
