"""manifold_topology: coordinate-free, label-free topological invariants of a point cloud.

Computes Betti numbers (b0, b1) and the persistence diagram of a point cloud in PCA-k space via
Vietoris-Rips persistent homology. Distinguishes a *loop* (weekdays, b1=1) from a *line*
(alphabet, b1=0) using geometry alone — the purest universality signal (no coordinates, no labels).

This is a *method* (interpretability primitive) — see ARCHITECTURE.md §3:
  - third-party imports only (ripser); no causalab.runner / causalab.analyses
  - no hyperparameter defaults
  - no disk I/O
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def significant_betti(diagram, *, rel_threshold, ref_scale) -> int:
    """Count finite features in a persistence ``diagram`` (one homology dim) whose lifetime
    (death - birth) exceeds ``rel_threshold`` × ``ref_scale``.

    ``ref_scale`` is an EXTERNAL scale (the point cloud's diameter), not the diagram's own
    max lifetime. This is essential: a relative-to-self threshold cannot distinguish "one real
    loop" from "all noise" — for a cloud with no true feature, the largest noise bar would always
    pass. A genuine loop persists over a meaningful fraction of the data extent; jitter loops do not.
    """
    if diagram is None or len(diagram) == 0:
        return 0
    d = np.asarray(diagram, dtype=float)
    life = d[:, 1] - d[:, 0]
    life = life[np.isfinite(life)]
    if life.size == 0:
        return 0
    return int((life > rel_threshold * ref_scale).sum())


def persistent_homology(points, *, maxdim, rel_threshold) -> dict:
    """Vietoris-Rips persistent homology of ``points`` [n,k].

    Returns {'diagrams': list-per-dim, 'betti': {0: int, 1: int, ...}}.
    b0 = number of connected components (infinite-death bars in H0).
    b_d (d>=1) = count of significant finite bars via ``significant_betti``, thresholded
    against the cloud diameter so noise loops in featureless clouds are not counted.
    """
    from ripser import ripser  # lazy: keeps the module importable without the backend

    X = _to_numpy(points)
    # external reference scale: the cloud diameter (max pairwise distance)
    ref_scale = float(np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1).max())
    dgms = ripser(X, maxdim=int(maxdim))["dgms"]
    betti: dict[int, int] = {}
    h0 = dgms[0]
    betti[0] = int(np.isinf(np.asarray(h0)[:, 1]).sum()) if len(h0) else 0
    for d in range(1, int(maxdim) + 1):
        betti[d] = significant_betti(dgms[d], rel_threshold=rel_threshold, ref_scale=ref_scale)
    return {"diagrams": [np.asarray(dg).tolist() for dg in dgms], "betti": betti}


def graph_betti(points, *, k) -> dict:
    """Betti numbers of the symmetric k-nearest-neighbour graph's 1-skeleton.

    For few-point, high-dimensional, clustered concept manifolds (e.g. 7 weekday centroids in R^8),
    Vietoris-Rips H1 is unreliable — the loop is born late and filled almost immediately. The k-NN
    graph captures the ring ordering directly: b1 = E - V + C (independent cycles), b0 = C (components).
    A clean cyclic concept (weekdays, k=2) gives b1=1; a line/diffuse concept gives b1=0. Returns
    {'b0', 'b1', 'n_edges'}.
    """
    X = _to_numpy(points)
    n = X.shape[0]
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    knn = np.argsort(D, axis=1)[:, : int(k)]
    edges = set()
    for i in range(n):
        for j in knn[i]:
            edges.add((min(i, int(j)), max(i, int(j))))  # symmetric: edge if i->j OR j->i
    parent = list(range(n))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb
    comps = len({_find(i) for i in range(n)})
    b1 = len(edges) - n + comps
    return {"b0": int(comps), "b1": int(max(0, b1)), "n_edges": int(len(edges))}


def betti_match(betti_A: dict, betti_B: dict) -> dict:
    """Do two models' same-concept manifolds agree on Betti numbers? Returns {'match', 'per_dim'}."""
    keys = sorted(set(betti_A) & set(betti_B))
    per_dim = {k: bool(betti_A[k] == betti_B[k]) for k in keys}
    return {"match": all(per_dim.values()) if per_dim else False, "per_dim": per_dim}
