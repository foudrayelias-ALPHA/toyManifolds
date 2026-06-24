---
name: cross_model_alignment
---

# Method spec: `cross_model_alignment`

(Canonical copy; full design rationale in `${SESSION_DIR}/plan/setup_method_cross_model_alignment.md`.)

## §1. Identity

Label-free alignment of two models' concept manifolds. Given each model's concept point cloud /
distance matrix (in its own PCA subspace), recover a correspondence using Gromov-Wasserstein OT
(headline) and orthogonal Procrustes (linear baseline), and score how interpretable that
correspondence is against null models. Labels never enter the alignment — only the post-hoc scorer.

## §2. Surface

- `gromov_wasserstein(D_A, D_B, *, epsilon, n_init, seed, p=None, q=None) -> dict` → `{coupling, gw_distance, converged}`
- `coupling_to_matching(coupling) -> np.ndarray` (Hungarian hard assignment)
- `procrustes_align(X_A, X_B, correspondence) -> dict` → `{R, residual, scale}`
- `label_recovery(matching, labels_A, labels_B, *, symmetry_group) -> dict` → `{raw, mod_symmetry, best_group_element}`
- `null_distribution(D_A, D_B, *, kind, n_samples, seed, gw_epsilon, gw_n_init) -> dict` → `{samples, mean, p5, p95}`

## §3. Dependencies

Third-party only: `numpy`, `ot` (POT, already a project dep), `scipy.linalg.orthogonal_procrustes`,
`scipy.optimize.linear_sum_assignment`. No `causalab.runner` / `causalab.analyses` imports.

## §4. Hyperparameters (no defaults — supplied by the analysis)

`epsilon` (entropic GW reg), `n_init` (GW restarts), `seed`, `symmetry_group` (`cyclic`/`none`),
null `kind` (`shuffle`/`random`), `n_samples`, `gw_epsilon`, `gw_n_init`.

## §5. Side effects

None. Returns in-memory dicts. No disk I/O.

## Notes

GW is invariant to relabelling, so the null destroys *geometry* (shuffles distance-matrix entries),
not labels. `mod_symmetry` quotients the dihedral group of the cycle — the identifiable phase quantity.
