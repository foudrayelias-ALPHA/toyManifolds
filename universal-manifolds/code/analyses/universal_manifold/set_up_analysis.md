---
name: universal_manifold
---

# Analysis spec: `universal_manifold`

(Canonical copy; full design in `${SESSION_DIR}/plan/setup_analysis_universal_manifold.md`.)

## Research question

Is the source model's concept manifold a low-distortion, unsupervised image of the target model's,
and does the recovered correspondence match the human concept? (Objective SC1–SC4 / H1–H4.)

## §1. Cross-root inputs (architectural deviation)

Reads TWO producer experiment_roots via explicit `source`/`target` blocks — NOT single-root
auto-discovery. Per side: `{root}/{variant}/subspace/{subspace}/layer_x_pos/L{layer}_{pos}/features/training_features.safetensors`
(key `features`). Per-class centroids = class-means (labels regenerated from the task; see first-run note).

## §2. Methods (§3 dependencies)

- session-local `methods.cross_model_alignment`: `gromov_wasserstein`, `coupling_to_matching`,
  `procrustes_align`, `label_recovery`, `null_distribution`.
- session-local `methods.manifold_topology`: `persistent_homology`, `betti_match`.
- shipped `causalab.methods.scores.isometry.compute_isometry_metrics` (distance-correlation).
- `causalab.io.artifacts` (save/load), `causalab.runner.helpers` (resolve_task, generate_datasets).

## §3. Hyperparameters (defaults in analysis.yaml, not code)

`gw.{epsilon,n_init,seed}`, `procrustes.correspondence`, `symmetry_group`, `topology.{maxdim,rel_threshold}`,
`nulls.{kinds,n_samples,seed}`, `granularity`, `target_variable`, `source/target.{experiment_root,variant,subspace,layer,pos}`, `pair_label`.

## §4. Outputs

`metrics.json`, `coupling.safetensors`, `visualization/coupling.png`, `metadata.json` under
`${experiment_root}/universal_manifold/${pair_label}/`.

## §5. Side effects / first-run validation

No model weights loaded (CPU). The per-example label loader assumes subspace `features` rows are in
regenerated train order; if the producer filtered to correct-only examples the length guard fires —
align the loader to the filtered indices on the first real run (no producer artifacts exist yet to test against).
