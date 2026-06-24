# Method spec: `cross_model_alignment`

Session-local reusable primitive (no Hydra, no I/O paths — pure functions on arrays/tensors).
Scaffold under `${SESSION_DIR}/code/methods/cross_model_alignment/`. Consumed by the
`universal_manifold` analysis.

## Purpose

Align two models' concept point clouds (each in its own ℝᵏ PCA subspace) **without labels**, and score
how universal / interpretable the alignment is. Headline = Gromov-Wasserstein; Procrustes = linear baseline.

## New dependencies (add via `uv add`)

- `pot` (Python Optimal Transport, import as `ot`) — Gromov-Wasserstein.
- (`scipy` for `scipy.linalg.orthogonal_procrustes`, `numpy` — assumed already present.)

## Public functions

```python
def gromov_wasserstein(D_A, D_B, p=None, q=None, epsilon=5e-3, n_init=10, seed=0) -> dict:
    """Entropic GW between two intra-space distance matrices (NO shared coords, NO labels).
    D_A: [m,m], D_B: [n,n] (m may ≠ n). Returns {coupling [m,n], gw_distance: float,
    converged: bool}. Run n_init random inits, keep the lowest-cost coupling."""

def procrustes_align(X_A, X_B, correspondence) -> dict:
    """Orthogonal Procrustes baseline. X_A:[m,k], X_B:[m,k] matched by `correspondence`
    (index array or the hard matching from a GW coupling). Returns {R [k,k], residual: float,
    scale: float}. Uses scipy.linalg.orthogonal_procrustes."""

def coupling_to_matching(coupling) -> np.ndarray:
    """Hard 1-1 assignment from a soft coupling (argmax per row, or Hungarian on -coupling)."""

def label_recovery(matching, labels_A, labels_B, symmetry_group=None) -> dict:
    """Fraction of source classes mapped to the same human label in the target.
    Returns {raw: float, mod_symmetry: float, best_group_element: ...}.
    For symmetry_group='cyclic' (n classes): take the max recovery over all n rotations ×
    2 reflections of the cycle — the 'up to symmetry' score that SC2 requires.
    For None: mod_symmetry == raw."""

def null_distribution(D_A, D_B, kind='shuffle', n_samples=200, seed=0) -> dict:
    """Null GW distances. kind='shuffle' permutes one side's row/col order before GW;
    kind='random_projection' replaces one cloud with a random orthogonal image of matched
    variance. Returns {samples: [n], mean, p5, p95}."""
```

## Notes / invariants

- Pure functions; accept numpy or torch (convert internally), return plain Python / numpy.
- Distance metric for `D_A`/`D_B` is Euclidean on the PCA-`k` features by default; keep it a parameter so the
  caller can pass geodesic distances later.
- Do **not** read labels inside `gromov_wasserstein` or `procrustes_align`'s correspondence search — labels enter
  **only** via `label_recovery` (post-hoc scoring). This enforces the "compute over human-imposed ideas" contract.
- Determinism: every randomized routine takes `seed`.

## Minimal tests (under `code/methods/cross_model_alignment/tests/` or a smoke check)

- Two identical clouds → GW distance ≈ 0, matching is identity (mod symmetry), label_recovery.mod_symmetry == 1.0.
- A rotated/relabeled copy of a cyclic cloud → mod_symmetry ≈ 1.0 even when raw < 1.0.
- Shuffle null mean ≫ the true-pair GW distance for the identical-cloud case.
