# Universal Manifold

`universal_manifold` answers: *is one model's concept manifold a low-distortion, unsupervised image
of another model's, and does the recovered correspondence match the human concept?* It is a
**cross-model** analysis — unlike every shipped analysis it reads **two** producer `experiment_root`s
(a source and a target model). For each side it loads the cached PCA-subspace feature cloud (from
`subspace`), groups it into per-class centroids, then aligns the two manifolds with Gromov-Wasserstein
optimal transport (label-free) and scores universality: near-isometry (distance-correlation, reusing
`causalab/methods/scores/isometry.py`), a Procrustes baseline, unsupervised label-recovery (raw +
up-to-cyclic-symmetry), Betti numbers (`manifold_topology`), and geometry-destroying null models. No
model weights are loaded — it runs on cached features (CPU).

It sits at the *end* of the pipeline, downstream of `subspace`/`activation_manifold` for two models.
Labels are used **only** in the post-hoc recovery score; alignment is label-free.

## Configuration

**Root config** (`causalab/configs/config.yaml`):
- `experiment_root` — output root; this analysis writes to `${experiment_root}/universal_manifold/${.pair_label}`.
- `seed` — referenced by `gw.seed` / `nulls.seed`.

**Module config** (`code/configs/analysis/universal_manifold.yaml`, `# @package universal_manifold`):

```yaml
_name_: universal_manifold
_subdir: ${.pair_label}                 # output subdir = the pair label
_output_dir: ${experiment_root}/universal_manifold/${._subdir}
pair_label: ???                         # e.g. weekdays_1b_8b (set by runner)
source: {experiment_root, variant, subspace, layer, pos}   # producer A (explicit, no auto-discovery)
target: {experiment_root, variant, subspace, layer, pos}   # producer B
target_variable: entity                 # class identity for centroids + label-recovery
granularity: [centroid, cloud]          # GW + distance-correlation at both levels
gw: {epsilon: 0.02, n_init: 10, seed: ${seed}}
procrustes: {correspondence: gw}        # or "labels" (supervised upper bound)
symmetry_group: cyclic                  # "none" for non-cyclic concepts (alphabet)
topology: {maxdim: 1, rel_threshold: 0.25}
nulls: {kinds: [shuffle, random], n_samples: 200, seed: ${seed}}
visualization: {figure_format: png}
```

## Outputs

Directory: `${experiment_root}/universal_manifold/${pair_label}/`

### Interpretation

- **`metrics.json`** — the headline. `centroid.distance_correlation` ≥ 0.70 and `centroid.nulls.shuffle.pvalue` < 0.05
  support universality (SC1). `centroid.label_recovery.mod_symmetry` ≥ 6/7 means geometry recovered the human
  concept up to the cycle's symmetry (SC2); `raw` is the absolute-phase bonus. `cloud.betti.match` with
  b₁=1 (weekdays) / b₁=0 (alphabet) is the topological signal (SC3). Compare same-concept vs cross-concept
  pairs for discriminant validity (SC4).
- **`visualization/coupling.png`** — GW coupling heatmap; a near-permutation (one bright cell per row) means a
  clean correspondence; diffuse mass means the manifolds did not align cleanly.

### Saved artifacts

| File | Format | Notes |
|---|---|---|
| `metrics.json` | JSON | all scalars: GW distance, distance-correlation, Procrustes residual, label-recovery, Betti, nulls |
| `coupling.safetensors` | safetensors | `centroid_coupling` [C,C] (+ `cloud_coupling` if cloud granularity) |
| `visualization/coupling.png` | png | GW coupling heatmap |
| `metadata.json` | JSON | resolved Hydra config snapshot (provenance) |
