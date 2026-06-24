"""cross_model_alignment: label-free alignment of two models' concept manifolds.

Aligns two point clouds / distance matrices (each in its own PCA subspace) using
Gromov-Wasserstein optimal transport (headline, label-free, dimension-agnostic) and
orthogonal Procrustes (linear baseline), then scores how interpretable the recovered
correspondence is (label-recovery, raw + up-to-symmetry) against null models.

This is a *method* (interpretability primitive) — see ARCHITECTURE.md §3:
  - imports only from third-party libs (no causalab.runner / causalab.analyses)
  - no hyperparameter defaults (the consuming analysis's Hydra config supplies them)
  - no disk I/O (the consuming analysis decides where results land)

Labelling contract: labels enter ONLY ``label_recovery`` (post-hoc scoring). The
alignment routines (``gromov_wasserstein``, ``procrustes_align``) never see labels.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import ot
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment


def _to_numpy(x: Any) -> np.ndarray | None:
    if x is None:
        return None
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def _gw_cost(C1: np.ndarray, C2: np.ndarray, T: np.ndarray) -> float:
    """Square-loss Gromov-Wasserstein cost of coupling ``T`` (small sizes only)."""
    diff = C1[:, None, :, None] - C2[None, :, None, :]
    return float(np.einsum("ijkl,ij,kl->", diff * diff, T, T))


def gromov_wasserstein(D_A, D_B, *, epsilon, n_init, seed, p=None, q=None) -> dict:
    """Entropic Gromov-Wasserstein between two intra-space distance matrices.

    NO shared coordinates, NO labels. ``D_A`` [m,m] and ``D_B`` [n,n] (m may differ
    from n). Runs ``n_init`` restarts (first from the product init, the rest from
    random feasible couplings) and keeps the lowest-cost coupling.

    Returns {'coupling': [m,n], 'gw_distance': float, 'converged': bool}.
    """
    C1 = _to_numpy(D_A)
    C2 = _to_numpy(D_B)
    # Scale-invariance: normalize each intra-space distance matrix to unit max so the entropic
    # regularization `epsilon` is comparable across spaces of different magnitude. Deep-layer
    # activations have large norms; without this, entropic GW underflows and returns a spurious 0.
    # This is also conceptually right — universality is about manifold *shape*, not absolute scale.
    _m1, _m2 = float(C1.max()), float(C2.max())
    if _m1 > 0:
        C1 = C1 / _m1
    if _m2 > 0:
        C2 = C2 / _m2
    m, n = C1.shape[0], C2.shape[0]
    p = np.full(m, 1.0 / m) if p is None else _to_numpy(p)
    q = np.full(n, 1.0 / n) if q is None else _to_numpy(q)
    rng = np.random.default_rng(seed)

    best: dict | None = None
    for it in range(max(1, int(n_init))):
        G0 = None
        if it > 0:
            if m == n and np.ptp(p) < 1e-12 and np.ptp(q) < 1e-12:
                # uniform square case (centroids): a random permutation plan is exactly feasible
                perm = rng.permutation(n)
                G0 = np.zeros((m, n))
                G0[np.arange(n), perm] = 1.0 / n
            else:
                # general case: marginal-feasible random init via Sinkhorn on a random cost
                G0 = ot.sinkhorn(p, q, rng.random((m, n)), reg=1.0, numItermax=500)
        try:
            T, log = ot.gromov.entropic_gromov_wasserstein(
                C1, C2, p=p, q=q, loss_fun="square_loss",
                epsilon=float(epsilon), G0=G0, log=True, verbose=False,
            )
        except TypeError:  # older POT without the G0 kwarg
            T, log = ot.gromov.entropic_gromov_wasserstein(
                C1, C2, p, q, "square_loss", float(epsilon), log=True, verbose=False,
            )
        gw = log.get("gw_dist")
        if gw is None:
            gw = _gw_cost(C1, C2, np.asarray(T)) if (m * n) <= 400 else float("nan")
        gw = float(gw)
        if best is None or gw < best["gw_distance"]:
            best = {"coupling": np.asarray(T), "gw_distance": gw, "converged": True}
    assert best is not None
    return best


def coupling_to_matching(coupling) -> np.ndarray:
    """Hard 1-1 assignment from a soft coupling (Hungarian on -coupling).

    Returns an int array ``matching`` of length m where ``matching[i]`` is the target
    index assigned to source row i.
    """
    T = _to_numpy(coupling)
    row, col = linear_sum_assignment(-T)
    matching = np.empty(T.shape[0], dtype=int)
    matching[row] = col
    return matching


def procrustes_align(X_A, X_B, correspondence) -> dict:
    """Orthogonal Procrustes baseline mapping source ``X_A`` onto target ``X_B``.

    ``X_A`` [m,k], ``X_B`` [n,k]; ``correspondence`` length m gives the target row for
    each source row. Returns {'R' [k,k], 'residual': float, 'scale': float} where R
    minimises ||A R - B||_F on mean-centred, correspondence-matched points.
    """
    XA = _to_numpy(X_A)
    XB = _to_numpy(X_B)
    idx = np.asarray(correspondence, dtype=int)
    XBm = XB[idx]
    A = XA - XA.mean(axis=0, keepdims=True)
    B = XBm - XBm.mean(axis=0, keepdims=True)
    R, scale = orthogonal_procrustes(A, B)
    resid = float(np.linalg.norm(A @ R - B) / (np.linalg.norm(B) + 1e-12))
    return {"R": np.asarray(R), "residual": resid, "scale": float(scale)}


def label_recovery(matching, labels_A: Sequence, labels_B: Sequence, *, symmetry_group) -> dict:
    """Does the unsupervised matching land on the human concept? (post-hoc; labels used HERE ONLY.)

    ``matching[i]`` = target index for source class i. ``labels_A``/``labels_B`` are the
    human labels in source/target index order. Returns {'raw', 'mod_symmetry',
    'best_group_element'}.

    - raw: fraction of source classes mapped to a target with the same label.
    - symmetry_group='cyclic': max recovery over the dihedral group of the n-cycle
      (all n rotations × 2 reflections) — the 'up to symmetry' score (SC2). A symmetric
      cycle admits no absolute phase, so this is the identifiable quantity.
    - symmetry_group in (None,'none'): mod_symmetry == raw.
    """
    matching = np.asarray(matching, dtype=int)
    m = matching.shape[0]
    labels_A = list(labels_A)
    labels_B = list(labels_B)
    raw = float(np.mean([labels_B[matching[i]] == labels_A[i] for i in range(m)]))

    if symmetry_group in (None, "none"):
        return {"raw": raw, "mod_symmetry": raw, "best_group_element": "identity"}
    if symmetry_group == "cyclic":
        label_pos_B = {lab: j for j, lab in enumerate(labels_B)}
        canon = np.array([label_pos_B[labels_A[i]] for i in range(m)], dtype=int)
        n = len(labels_B)
        best, best_g = -1.0, None
        for refl in (False, True):
            base = (-canon) % n if refl else canon
            for k in range(n):
                g = (base + k) % n
                score = float(np.mean(matching == g))
                if score > best:
                    best, best_g = score, ("reflect" if refl else "rotate", int(k))
        return {"raw": raw, "mod_symmetry": best, "best_group_element": best_g}
    raise ValueError(f"unknown symmetry_group: {symmetry_group!r}")


def null_distribution(D_A, D_B, *, kind, n_samples, seed, gw_epsilon, gw_n_init) -> dict:
    """Null GW distances for the 'is the true alignment better than chance geometry?' test.

    NOTE: GW is invariant to relabelling, so a symmetric row/col permutation of D_B is
    NOT a valid null. Instead we destroy the *geometry*:
      - kind='shuffle': permute the off-diagonal distance entries of D_B (breaks the metric).
      - kind='random' / 'random_projection': resample entries uniformly in D_B's range.
    Then recompute GW(D_A, D_null). Returns {'samples', 'mean', 'p5', 'p95'}.
    """
    C1 = _to_numpy(D_A)
    C2 = _to_numpy(D_B)
    n = C2.shape[0]
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(n, k=1)
    vals = C2[iu]
    samples = []
    for s in range(int(n_samples)):
        if kind == "shuffle":
            newv = rng.permutation(vals)
        elif kind in ("random", "random_projection"):
            newv = rng.uniform(float(vals.min()), float(vals.max()), size=vals.shape)
        else:
            raise ValueError(f"unknown null kind: {kind!r}")
        Cn = np.zeros((n, n))
        Cn[iu] = newv
        Cn = Cn + Cn.T
        gw = gromov_wasserstein(
            C1, Cn, epsilon=gw_epsilon, n_init=max(1, int(gw_n_init)), seed=int(seed) + s
        )["gw_distance"]
        samples.append(gw)
    samples = np.asarray(samples, dtype=float)
    return {
        "samples": samples.tolist(),
        "mean": float(np.nanmean(samples)),
        "p5": float(np.nanpercentile(samples, 5)),
        "p95": float(np.nanpercentile(samples, 95)),
    }
