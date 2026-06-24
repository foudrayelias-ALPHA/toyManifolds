# Analysis spec: `universal_manifold`

Session-local Hydra analysis. Scaffold under `${SESSION_DIR}/code/analyses/universal_manifold/` (main.py +
README.md) with config `${SESSION_DIR}/code/configs/analysis/universal_manifold.yaml` (`# @package universal_manifold`).
Wraps the `cross_model_alignment` and `manifold_topology` methods and reuses the shipped isometry scorer.

## Research question

Is the **source** model's concept manifold a low-distortion, unsupervised image of the **target** model's,
and does the recovered correspondence match the human concept? (Objective SC1–SC4, H1–H4.)

## Architectural deviation — cross-root inputs

Unlike shipped analyses (single `experiment_root`, auto-discovery), this analysis reads **two** producer roots.
Provide them as explicit config blocks; **do not** auto-discover a single root for the manifold inputs:

```yaml
# @package universal_manifold
analysis:
  _name_: universal_manifold
  _output_dir: ${experiment_root}          # synthetic root: .../artifacts/universal_manifold/<pair_label>

  source:
    experiment_root: ???                    # e.g. agent_logs/<sess>/artifacts/natural_domains_arithmetic/llama32_1b_instruct
    variant: weekdays                       # appended → .../weekdays
    subspace: pca_k8                        # subspace _subdir to read training_features + labels from
    activation_manifold: pca_k8/spline_s0.0 # for centroids + intrinsic map (viz / interpretability lens)
  target:
    experiment_root: ???                    # e.g. .../llama31_8b
    variant: weekdays
    subspace: pca_k8
    activation_manifold: pca_k8/spline_s0.0

  granularity: [centroid, cloud]            # GW + distance-correlation at both levels
  distance_metric: euclidean

  gw:
    epsilon: 5.0e-3
    n_init: 10
    seed: 0
  procrustes:
    correspondence: gw                      # 'gw' (use GW hard matching) or 'labels' (supervised upper-bound)
  symmetry_group: cyclic                    # for label-recovery mod-symmetry (weekdays); 'none' for alphabet
  topology:
    maxdim: 1
    rel_threshold: 0.25
  nulls:
    kinds: [shuffle, random_projection]
    n_samples: 200
    seed: 0
  visualization:
    figure_format: png
```

`pair_label` (e.g. `weekdays_1b_8b`) is the last path segment of `experiment_root`, set by the runner config.

## Inputs read (per side)

- `{root}/{variant}/subspace/{subspace}/training_features.safetensors` — per-example points in ℝᵏ + class labels.
- `{root}/{variant}/activation_manifold/{activation_manifold}/…` — per-class centroids + intrinsic coords (viz + intrinsic-map lens).

## Computation

1. Build `D_A`, `D_B` (intra-space distance matrices) at centroid and cloud granularity.
2. **GW** (`cross_model_alignment.gromov_wasserstein`) → coupling + `gw_distance` (both granularities).
3. **Distance-correlation** — reuse `causalab.methods.scores.isometry.compute_isometry_metrics(D_A, D_B)` (Pearson) for the near-isometry metric.
4. **Procrustes** baseline (`procrustes_align`) using the GW hard matching (and, separately, the label matching as a supervised upper bound).
5. **Label recovery** (`label_recovery`, `symmetry_group=cyclic`) → raw + mod-symmetry accuracy. **Labels used here only.**
6. **Topology** (`manifold_topology.persistent_homology` on each cloud) → `b₀,b₁` per model + `betti_match`.
7. **Nulls** (`null_distribution`) → shuffle + random-projection GW distributions; report the true pair's percentile/p-value.

## Outputs (to `${experiment_root}` = `.../universal_manifold/<pair_label>/`)

- `metrics.json` — `{gw_distance: {centroid, cloud}, distance_correlation: {centroid, cloud},
  procrustes_residual: {gw_corr, label_corr}, label_recovery: {raw, mod_symmetry, best_group_element},
  betti: {A, B, match}, nulls: {shuffle: {mean,p5,p95,pvalue}, random_projection: {...}}, provenance: {source, target}}`.
- `coupling.pt` — the GW coupling matrix (centroid + cloud).
- `visualization/{coupling.png, manifolds_aligned_3d.html, persistence_A.png, persistence_B.png, isometry_scatter.png}`.

## Pre-flight gate

Both sides' `training_features.safetensors` exist and are non-empty; `metrics.json` writes finite
`gw_distance` and `distance_correlation`.

## Runtime

~2–8 min, **CPU** (no model weights loaded — operates purely on cached features/manifolds).

## Notes

- No model forward passes — like `activation_manifold` with `skip_decoding_eval`, build a lite pipeline or none.
- Keep the labels-only-for-scoring contract: only step 5 touches `labels`. Steps 2–4, 6–7 are label-free.
