---
name: manifold_topology
---

# Method spec: `manifold_topology`

(Canonical copy; full rationale in `${SESSION_DIR}/plan/setup_method_manifold_topology.md`.)

## §1. Identity

Coordinate-free, label-free topological invariants of a point cloud in PCA-k space: Betti numbers
(b0, b1) and the persistence diagram via Vietoris-Rips persistent homology. Separates a loop
(weekdays, b1=1) from a line (alphabet, b1=0) using geometry alone.

## §2. Surface

- `persistent_homology(points, *, maxdim, rel_threshold) -> dict` → `{diagrams, betti}`
- `significant_betti(diagram, *, rel_threshold) -> int`
- `betti_match(betti_A, betti_B) -> dict` → `{match, per_dim}`

## §3. Dependencies

Third-party only: `numpy`, `ripser` (lazy import; `persim` available for diagram plots at the analysis
layer). No `causalab.runner` / `causalab.analyses` imports.

## §4. Hyperparameters (no defaults)

`maxdim` (homology dims; 1 for b0/b1), `rel_threshold` (relative-lifetime significance cut for finite bars).

## §5. Side effects

None. Returns in-memory dicts. No disk I/O.

## Notes

Operate on the per-example point cloud (subspace `training_features`), not the 7 centroids (too sparse).
b0 = count of infinite-death H0 bars (connected components); b_{d>=1} = significant finite bars.
