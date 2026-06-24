# Issues — universal-manifolds

## Open — validate on first real run (no producer artifacts exist yet)

### 1. [RESOLVED 2026-06-21] `universal_manifold` label/feature loading
Original concern: regenerating labels could misalign with correct-only-filtered features. **Resolved**
after inspecting real producer output: the subspace dir is `subspace/pca_k8/<target_variable>/` and ships
a co-located `train_dataset.json` (the exact dataset that produced `features/training_features.safetensors`,
same order). `main.py::_load_labels` now reads labels from that file — no regeneration, no ordering
assumption, and it also fixes the cross-concept discriminant pair (each side reads its own concept's labels).
Real layout also corrected `_load_cloud` (key `features`, shape `[n, 8]`).

### 2. `compute_isometry_metrics` return-key not pinned
`_iso_scalar` extracts the distance-correlation scalar best-effort (tries `pearson_r`/`isometry`/… then
first float). Confirm the actual key from `causalab/methods/scores/isometry.py` on first run and pin it.

### 3. GW inner-Sinkhorn convergence at small epsilon (minor)
Entropic GW emits a benign "Sinkhorn did not converge" warning at very small `epsilon`. Analysis default
is `gw.epsilon: 0.02` (stable). If the centroid coupling looks over-diffuse, lower epsilon; if warnings/
instability, raise it or increase inner iters.

### 4. [RESOLVED 2026-06-23] GW was not scale-invariant → spurious 0 at deep layers
`gromov_wasserstein` ran entropic GW on raw distance matrices with fixed `epsilon=0.02`. Deep-layer
activations have large norms (8B L16 centroid-dist max≈6.0 vs 1B L0 ≈1.45), so the entropic term
underflowed and returned GW≈0 with null_mean≈0 (impossible for genuinely different manifolds). **Fixed**:
normalize each distance matrix to unit max before the solve (also conceptually correct — universality is
about shape, not absolute scale). All reported numbers are post-fix. Superseded issue #3.

### 5. [RESOLVED-by-substitution 2026-06-23] Topology (SC3/H3) — Betti is unreliable at this scale
Tried two Betti operationalizations, both unreliable for 7-point/8-D clustered centroids: (a) Vietoris–Rips PH
gives a real but vanishing H1 bar (life/diam≈0.05 — the loop fills instantly in 8-D); (b) symmetric 2-NN-graph
Betti over-counts cycles (b₁≈3–4 weekdays, ≈12–15 alphabet — scales with class count, not topology). **Resolution**:
use the **nearest-neighbour ring-adjacency (cyclic-NN) fraction** as the structural metric — 1.00 (weekdays) vs
0.35 (alphabet), clean and interpretable. Both Betti variants retained in metrics.json as informative-only.
TODO on promotion: add `ring_adjacency` as a first-class method.

### 6. [OPEN] GW shuffle-null is too permissive at small class counts
With only 7 weekday centroids, entropic GW finds a low-cost coupling to almost any 7-point target, so the
entry-shuffle null loses significance at deep layers (p=0.175/0.487). The discriminant (same vs cross concept)
is the reliable test at small N. Consider a Procrustes-residual permutation test or random-manifold nulls.

## Resolved
- #1 label/feature loading (real layout: subspace/pca_k8/<tv>[/layer_x_pos/L{n}_{pos}]/features + co-located train_dataset.json).
- #2 isometry return-key is `pearson_r` (pinned in `_iso_scalar`).
- #3 → superseded by #4 (scale-invariant GW).
- #4 GW scale-invariance.
- Kaggle 8B: first run failed (Hydra log → read-only /kaggle/input cwd); fixed with cwd=/kaggle/working; v2 clean.
