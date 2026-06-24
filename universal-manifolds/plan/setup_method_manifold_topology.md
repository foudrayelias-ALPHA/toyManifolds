# Method spec: `manifold_topology`

Session-local reusable primitive (no Hydra, no I/O). Scaffold under
`${SESSION_DIR}/code/methods/manifold_topology/`. Consumed by the `universal_manifold` analysis to
provide a **coordinate-free, label-free** universality signal (SC3 / H3).

## Purpose

Compute topological invariants of a point cloud in PCA-`k` space: Betti numbers `b₀, b₁` (and the
persistence diagram) via persistent homology. Distinguishes a **loop** (weekdays, `b₁=1`) from a **line**
(alphabet, `b₁=0`) using geometry alone.

## New dependencies (add via `uv add`)

- `ripser` + `persim` (preferred — lightweight Vietoris–Rips persistent homology + diagram plotting),
  **or** `gudhi` if `ripser` wheels are unavailable on the run platform. Pick one; document which.

## Public functions

```python
def persistent_homology(points, maxdim=1) -> dict:
    """Vietoris–Rips persistence on points [n,k]. Returns {diagrams: list per dim,
    betti: {0: int, 1: int}, persistences: {...}}. Betti number per dim = count of
    features whose lifetime exceeds a significance threshold (see below)."""

def significant_betti(diagram, dim, rel_threshold=0.25) -> int:
    """Count features in `diagram` (dim) whose (death - birth) lifetime exceeds
    rel_threshold × the max lifetime at that dim. Robust to short-lived noise bars."""

def betti_match(betti_A, betti_B) -> dict:
    """{match: bool, per_dim: {0: bool, 1: bool}} — do two models' same-concept manifolds agree?"""
```

## Notes / invariants

- Operate on the **per-example point cloud** (subspace `training_features`), never on the 7 centroids — too
  sparse for homology.
- Normalize/standardize the cloud (zero-mean, unit-variance per the subspace pipeline) before Rips so the
  threshold is scale-comparable across models.
- `b₁` significance via `significant_betti` (relative-lifetime threshold), not raw bar count — a circle of
  noisy samples produces one long `H₁` bar plus short noise bars.
- Determinism: persistent homology is deterministic given the cloud; record library + version in any metadata.

## Minimal smoke check

- Points sampled on a noisy circle → `betti == {0:1, 1:1}`.
- Points sampled on a noisy line segment → `betti == {0:1, 1:0}`.
