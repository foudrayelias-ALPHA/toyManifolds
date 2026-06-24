# PLAN — Universal Manifolds Across Models (v1: Llama 1B → 8B, weekdays)

Companion to `RESEARCH_OBJECTIVE.md`. Read by `/run-experiment` (executes) and `/interpret-experiment`
(interpretation lens). Sections §B–§F per `PLAN_TEMPLATE.md`.

---

## §B. Causal model & dataset

**Task:** `natural_domains_arithmetic` — package at `causalab/tasks/natural_domains_arithmetic/`
- **Status:** `exists`. Two domains used, both already shipped as task configs:
  - `causalab/configs/task/natural_domains_arithmetic_weekdays.yaml` (variant `weekdays`) — **primary**
  - `causalab/configs/task/natural_domains_arithmetic_alphabet.yaml` (variant `alphabet`) — **discriminant control**
- No `/setup-task` needed.

### Causal variables (shared DAG: `(entity, number) → result → raw_output`)

| Name | Type | Cardinality (weekdays / alphabet) | Value space |
|---|---|---|---|
| `entity` | categorical, **cyclic** (weekdays) / ordinal, non-cyclic (alphabet) | 7 / 25 | weekdays: `Monday…Sunday`; alphabet: `A…Y` |
| `number` | ordinal | 7 / 4 | word-form `"one"…"seven"` (wd, cyclic) / `"one"…"four"` (alpha) |
| `result` | categorical, **cyclic** (wd, mod 7) / ordinal interval (alpha) | 7 / 22 | wd: a weekday; alpha: a letter `E…Z` |

- **weekdays** `result = (entityidx + number) mod 7` → a **7-cycle** (period 7, `b₁=1`).
- **alphabet** `result = chr(ord(entity) + number)`, no wrap (out-of-range pairs filtered) → a **line/interval** (`b₁=0`). This topological contrast is the discriminant test (SC3, H3/H4).

### Target variable — **LOCKED: `entity`** (checkpoint 2026-06-21)

- **v1 uses `entity`** (the day-of-week concept as *encoded from the input*). Rationale: it is encoded
  by both models regardless of arithmetic ability, maximizing the chance the 1B model yields a clean 7-cycle
  — this is the dominant risk to v1 (see §E). `entity` is a pure, scale-robust concept manifold.
- **Immediate extension: `result`** (the *computed* answer manifold — the repo's "weekday geometry" precedent).
  More interesting universality claim (does the *computation*'s geometry universalize?), but gated on the 1B
  model actually solving the arithmetic. Run after v1 on `entity` succeeds, or instead of it if the user
  prefers the computed-concept framing and accepts the 1B-accuracy risk.

### Mechanism summary

- `result = f(entity, number)` — weekdays: add `number` to `entity` mod 7; alphabet: shift letter by `number`, no wrap.
- `raw_output = output_prefix + result`.

### Expected behavior (golden-path; model must get these right for `result`-targeted runs)

```
weekdays: "Q: What day is three days after Monday?\nA:"           → " Thursday"
weekdays: "Q: What day is two days after Saturday?\nA:"           → " Monday"     (week-wrap)
alphabet: "...Starting at letter C, we increment by 2. The result is letter"  → " E"
```

For `entity`-targeted runs, no arithmetic competence is required — only that the model encodes the input day.

### Counterfactual generator

| Setting | Value | Why |
|---|---|---|
| `task.resample_variable` | `all` (task-config default) | centroid-mode locate; CF resamples every input. No pairwise needed in v1. |
| `locate.mode` | `centroid` (default) | meaningful under `resample_variable: all` (ARCHITECTURE §5 / ANALYSIS_GUIDE pitfalls). |

> Per ANALYSIS_GUIDE "Common Pitfalls": `pairwise` would require `resample_variable` = the single localized
> variable. We stay in `centroid` mode, so `resample_variable: all` is correct.

### Dataset sizing (from task config; unchanged)

```yaml
task:
  n_train: 1000        # capped by enumerate_all
  n_test: 50
  enumerate_all: true  # weekdays: 7×7 = 49 combos; alphabet: 25×4 = 100 (minus out-of-range)
  balanced: false
```

Rationale: the combinatorial space is tiny, so we exhaust it. Manifold fitting uses per-class centroids
(7 weekday classes / 22 alphabet classes) and the per-example point cloud (≤49 / ≤100 points).

---

## §C. Neural surface

### Models

| Model | Config | Hidden dim | Layers | Why |
|---|---|---|---|---|
| `meta-llama/Llama-3.2-1B-Instruct` | `model: llama32_1b_instruct` | 2048 | 16 | small end of the scale ladder |
| `meta-llama/Llama-3.1-8B` | `model: llama31_8b` | 4096 | 32 | large end; same family → highest-probability clean v1 |

Both `dtype: bfloat16`, `device: auto`, `slurm.gpus: 1`. After PCA to a fixed `k_features` (default **8**),
both manifolds live in **ℝ⁸** → Procrustes (orthogonal 8×8) applies directly; GW does not even require matched
dims, so the framework generalizes to the deferred gpt2 (768-dim) contrast unchanged.

### Tokenization-check predictions

- Each weekday (`" Monday"…" Sunday"`) and each letter (`" E"…" Z"`) should be a **single token** under both
  Llama tokenizers (shared BPE family). Verified by the standard token-alignment check at run time; `GET_RESULT_TOKEN_PATTERN`
  already sums mass over `[" Monday","Monday"," monday"]`-style variants, so minor spacing differences are absorbed.
- 1B-Instruct uses a chat-tuned tokenizer but the same vocab; the bare-completion template should still align.

### Compute budget (estimates; small N)

| Phase | Where | Wall time (per producer run) | GPU |
|---|---|---|---|
| baseline | inline | ~1–3 min | 1 |
| locate (coarse layer scan) | inline | 1B ~3–6 min / 8B ~15–35 min | 1 |
| subspace (PCA) | inline | ~1 min | 1 |
| activation_manifold (spline, `skip_decoding_eval: true`) | inline | ~1–2 min | 1 |
| **producer total** | | 1B ~10 min / 8B ~25–45 min | 1 |
| universal_manifold (GW + Procrustes + persistent homology + nulls) | inline | ~2–8 min | **CPU** |

4 producer runs (weekdays×{1B,8B}, alphabet×{1B,8B}) + 3 aligner runs ≈ **~1.5–2.5 GPU-hours total**.

### Hardware constraints (LOCKED: Kaggle T4 headless)

- **Primary target: Kaggle headless GPU (T4, 16 GB).** 8B in bf16 ≈ 16 GB — fits a T4 but tight: use small
  inference `batch_size` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. 1B is trivial. Aligner is CPU-only.
  Watch the **P100 sm_60 torch gotcha** (per global CLAUDE.md): detect P100 and reinstall a cu121 torch build
  before importing torch, else `no kernel image` errors.
- One concurrent GPU session on free tier; the 4 producers run sequentially, well within weekly quota.
- If a slurm cluster is available instead, dispatch with `--slurm` (`slurm.gpus: 1` resolves automatically).

---

## §D. Analysis-chain DAG

Two stages: **(1) per-(model, concept) producers** — the shipped pipeline, run 4× — and **(2) a custom
cross-model aligner** that consumes pairs of producer outputs.

### DAG diagram

```
 PRODUCERS (run 4×: {weekdays, alphabet} × {1B, 8B})
   baseline ──► locate ──► subspace ──► activation_manifold
                              │                 │
                              │ (point cloud)   │ (centroids + intrinsic map)
                              ▼                 ▼
 ALIGNER (custom, run 3×: reads TWO producer roots)
                     universal_manifold
                       ├─ Gromov-Wasserstein coupling + distance   (headline)
                       ├─ Procrustes residual                       (linear baseline)
                       ├─ distance-correlation  (reuse isometry.py) (near-isometry)
                       ├─ persistent homology / Betti  (manifold_topology) (topology)
                       ├─ unsupervised label-recovery (raw + mod symmetry)
                       └─ null controls (shuffle correspondence, random projection)
```

### Producer nodes (shipped analyses — one card covers all 4 runs; knobs identical)

#### Node P1: `baseline`
- **Scoped question:** does each model produce well-formed output distributions over the concept classes (and, for `result` runs, solve the task)?
- **Upstream:** none.
- **Downstream:** `${experiment_root}/baseline/{accuracy.json, per_class_output_dists.safetensors, counterfactual_sanity.json}`.
- **Non-default knobs:** none (task config supplies sizing).
- **Pre-flight gate:** `accuracy.json` — for `entity` runs, `per_class_output_dists` exists and counterfactual_sanity passes (accuracy not required, since `entity` is input-encoded). For `result` runs, `accuracy ≥ 0.20`; **if the 1B model misses this on `result`, fall back to `entity` (see §E).**
- **Runtime:** ~1–3 min / 1 GPU.

#### Node P2: `locate`
- **Scoped question:** which `(layer, token_position)` cell most encodes the target variable in this model?
- **Method:** `interchange` (fast, no training).
- **Upstream:** reuses `baseline` output dists (auto).
- **Downstream:** `${experiment_root}/locate/interchange/{variable}/{results.json, heatmap.pdf}`.
- **Non-default knobs:** `layers`: 1B `[0,2,4,6,8,10,12,14]`, 8B `[0,4,8,12,16,20,24,28]` (coarse scan; cheap given small N). `mode: centroid`.
- **Pre-flight gate:** `{variable}/results.json` has ≥ 1 cell with `KL_drop ≥ 0.3`. Else stop this producer.
- **Runtime:** 1B ~3–6 min / 8B ~15–35 min / 1 GPU.

#### Node P3: `subspace`
- **Scoped question:** what `k`-dim subspace at the located cell carries the variable? (This is the point cloud the aligner consumes.)
- **Method:** `pca` (unsupervised — geometry-driven, per the no-human-imposition philosophy; **not** DAS, which is supervised).
- **Upstream:** `subspace.layers: null` → auto-resolves `best_cell` from `locate/interchange/{variable}/results.json`.
- **Downstream:** `${experiment_root}/subspace/pca_k8/{rotation.pt, training_features.safetensors, metadata.json, visualization/features_3d.html}`.
- **Non-default knobs:** `method: pca`, `k_features: 8` (≈ 2–3× the 7 weekday classes; same `k` both models so the aligned spaces match dim — **checkpoint knob**).
- **Pre-flight gate:** `metadata.json` `reconstruction_kl ≤ 1.0`.
- **Runtime:** ~1 min / 1 GPU.

#### Node P4: `activation_manifold`
- **Scoped question:** what is the smooth low-D geometry through the per-class centroids in that subspace (centroids + intrinsic coordinates for the aligner's interpretability lens + visualization)?
- **Method:** `spline` (TPS).
- **Upstream:** `subspace: null` → auto-discovers `subspace/pca_k8/`.
- **Downstream:** `${experiment_root}/activation_manifold/pca_k8/spline_s0.0/{manifold_spline/ckpt_final.pt, models/.../, visualization/manifold_3d.html, metadata.json}`.
- **Non-default knobs:** `smoothness: 0.0`; `manifold_intrinsic_coords: pca` (**geometry-driven, not `parameter`** — avoids imposing the known cyclic embedding); `skip_decoding_eval: true` (no model weights needed; fast).
- **Pre-flight gate:** `manifold_spline/` checkpoint exists and fit metadata reports finite ambient/intrinsic dims.
- **Runtime:** ~1–2 min / 1 GPU.

### Aligner node (custom — not yet implemented)

#### Node A1: `universal_manifold`  *(custom — out of scope of shipped `causalab/analyses/`)*
- **Scoped question:** is the source model's concept manifold a low-distortion, unsupervised image of the target model's, and does the recovered correspondence match the human concept?
- **Upstream (explicit, cross-root — NOT single-root auto-discovery):** two producer roots' `subspace/pca_k8/training_features.safetensors` (+ class labels) and `activation_manifold/pca_k8/spline_s0.0/` (centroids, intrinsic map). Passed as explicit `universal_manifold.source` / `.target` config blocks `{experiment_root, variant}` — see `setup_analysis_universal_manifold.md`.
- **Downstream:** `${SESSION_DIR}/artifacts/universal_manifold/{pair_label}/{metrics.json, coupling.pt, visualization/{coupling.png, manifolds_aligned_3d.html, persistence_A.png, persistence_B.png, isometry_scatter.png}}`.
- **Non-default knobs (new analysis — defaults defined in its own config):** `gw.epsilon` (entropic reg), `gw.n_init` (≥ 10, escape local minima), `granularity: [centroid, cloud]`, `symmetry_group: cyclic` (for the mod-symmetry label-recovery score), `nulls: [shuffle, random_projection]`.
- **Needed implementations:**
  - method `cross_model_alignment` — GW (POT `ot.gromov`), orthogonal Procrustes (`scipy.linalg.orthogonal_procrustes`), coupling→matching, label-recovery (raw + mod symmetry group), null models. Spec: `plan/setup_method_cross_model_alignment.md`.
  - method `manifold_topology` — persistent homology / Betti numbers (`ripser`+`persim` or `gudhi`). Spec: `plan/setup_method_manifold_topology.md`.
  - analysis `universal_manifold` — Hydra entry point wrapping both methods, reusing `causalab/methods/scores/isometry.py::compute_isometry_metrics` for distance-correlation. Spec: `plan/setup_analysis_universal_manifold.md`.
  - **New deps:** `pot` (import `ot`), `ripser`+`persim` (or `gudhi`); `scipy`/`numpy` assumed present. `/setup-methods` adds them via `uv add`.
- **Pre-flight gate:** both producer roots contain a non-empty `subspace/pca_k8/training_features.safetensors`; `metrics.json` writes finite `gw_distance` and `distance_correlation`.
- **Spec paths:** see "Needed implementations" above.
- **Runtime:** ~2–8 min / CPU.

### Aligner invocations (3 pairs)

| `pair_label` | source | target | Tests |
|---|---|---|---|
| `weekdays_1b_8b` | weekdays-1B | weekdays-8B | **primary** — SC1, SC2, SC3 (b₁=1 both), H1, H2, H3 |
| `alphabet_1b_8b` | alphabet-1B | alphabet-8B | same-concept #2 (non-cyclic) — SC3 (b₁=0 both) |
| `weekdays1b_alphabet8b` | weekdays-1B | alphabet-8B | **discriminant** — SC4, H4 |

### Cross-analysis post-steps

None (single target variable per producer → no `variable_localization_heatmap`).

---

## §E. Risk register & contingency

### Pitfalls active for this plan

- **1B can't do the arithmetic (dominant v1 risk).** If targeting `result`, the 1B-Instruct model may fail
  weekday arithmetic → no clean `result` manifold. **Mitigation:** v1 targets `entity` (input-encoded, no
  arithmetic) by default; `result` is the gated extension.
- **Symmetry-group ambiguity (SC2).** A perfectly symmetric 7-cycle has dihedral symmetry → absolute phase
  (Monday↦Monday) is geometrically unidentifiable. **Mitigation:** SC2 scored *modulo* the symmetry group;
  raw phase reported as a bonus, expected to succeed only if the empirical manifold is irregular enough to break symmetry.
- **GW local minima.** Entropic GW is non-convex. **Mitigation:** `gw.n_init ≥ 10`, report best; sanity-check
  against the shuffle null.
- **Tiny centroid set (7 points).** Centroid-level GW is a 7↔7 problem — clean but symmetry-prone, and too
  sparse for topology. **Mitigation:** run GW at both `centroid` and `cloud` granularity; compute persistent
  homology on the point cloud, not centroids.
- **Cross-root architectural deviation.** Shipped analyses assume one `experiment_root`; the aligner reads two.
  **Mitigation:** explicit `source`/`target` config blocks (no auto-discovery); outputs routed to a synthetic
  `universal_manifold/{pair_label}` root. Documented in the setup-analysis spec.
- **`resample_variable` × `locate.mode`.** Not active — we use `centroid` mode with `resample_variable: all` (correct combo).
- **New dependencies.** `pot`, `ripser`/`gudhi` must install cleanly in the run environment (Kaggle/cluster). Verify in `/setup-methods`.

### Per-step contingency

| Node | If pre-flight fails, then |
|---|---|
| `baseline` (result runs) | 1B accuracy < 0.20 → **switch target to `entity`** and re-run; do not chase 1B arithmetic in v1. |
| `baseline` (entity runs) | malformed dists → revisit template/tokenization; stop chain. |
| `locate` | no cell with `KL_drop ≥ 0.3` → widen layer scan / check token positions; do not proceed to `subspace`. |
| `subspace` | `reconstruction_kl > 1.0` → raise `k_features` (8→16); re-fit. |
| `activation_manifold` | degenerate fit → raise `smoothness`; inspect `manifold_3d.html`. |
| `universal_manifold` | GW distance inside shuffle-null bulk for the **primary** pair → report as a *negative* universality result (a real, publishable bound), and check whether `entity`/`result` or `k_features` choice changed it before concluding. |

---

## §F. Outputs of the plan itself

### Runner config(s)

Not a classic `_subdir` sweep (members differ by **task/model**, which already separate `experiment_root`, and by
the cross-root aligner). **No `sweep_id`.** Seven configs under `${SESSION_DIR}/code/configs/runners/universal_manifolds/`:

Producers (chain `baseline → locate → subspace → activation_manifold`):
- `um_weekdays_1b.yaml`  (task weekdays, model llama32_1b_instruct)
- `um_weekdays_8b.yaml`  (task weekdays, model llama31_8b)
- `um_alphabet_1b.yaml`  (task alphabet, model llama32_1b_instruct)
- `um_alphabet_8b.yaml`  (task alphabet, model llama31_8b)

Aligners (analysis `universal_manifold` only; explicit source/target):
- `um_align_weekdays_1b_8b.yaml`
- `um_align_alphabet_1b_8b.yaml`
- `um_align_weekdays1b_alphabet8b.yaml`

### Output routing (per CONVENTIONS "Output routing")

- Producers: `--experiment-root agent_logs/<session>/artifacts/natural_domains_arithmetic/<model>`; the task
  `variant` (`weekdays`/`alphabet`) is auto-appended → `…/<model>/<variant>`.
- Aligners: `--experiment-root agent_logs/<session>/artifacts/universal_manifold/<pair_label>` (synthetic root;
  source/target read from explicit config).

### Overwrite-hazard check

- Producers write to distinct `…/{model}/{variant}/` roots — **no collision** (model and variant both differ).
- Within a producer, each analysis owns its `_subdir` (`pca_k8`, `spline_s0.0`) — no intra-run collision.
- Aligners write to distinct `universal_manifold/{pair_label}/` — no collision.
- No shared `locate/results.json` across differing `target_variables` (single target per producer). ✓

### Expected artifact tree

```
agent_logs/<session>/artifacts/
├── natural_domains_arithmetic/
│   ├── llama32_1b_instruct/
│   │   ├── weekdays/{baseline, locate/interchange/<var>, subspace/pca_k8, activation_manifold/pca_k8/spline_s0.0}/…
│   │   └── alphabet/{…same…}/…
│   └── llama31_8b/
│       ├── weekdays/{…}/…
│       └── alphabet/{…}/…
└── universal_manifold/
    ├── weekdays_1b_8b/{metrics.json, coupling.pt, visualization/…}
    ├── alphabet_1b_8b/{…}
    └── weekdays1b_alphabet8b/{…}
```

### Hand-off

1. **Before any run:** `/setup-methods` (scaffold `cross_model_alignment`, `manifold_topology`, add deps) then
   `/setup-analyses` (scaffold `universal_manifold` + its config). Both read the spec paths in §D.
2. `/run-experiment` — materializes the 7 runner configs and executes (producers first, then aligners).
3. `/interpret-experiment` — reads artifacts + this plan → `result/REPORT.md`.

---

## Review checkpoint — decisions (LOCKED 2026-06-21)

1. **Hypotheses + success criteria** (RESEARCH_OBJECTIVE.md SC1–SC4 / H1–H4) — **approved.**
2. **Target variable** — **`entity`** (scale-robust, clean concept). `result` deferred to extension.
3. **`k_features`** — **8** for both models (both manifolds in ℝ⁸).
4. **Scope** — **approved:** 4-producer + 3-aligner v1 (weekdays primary, alphabet discriminant). gpt2 / 70B / months / `result` / joint-canonical-manifold = later extensions.
5. **Compute target** — **Kaggle T4 headless** (push producers as GPU kernels; aligner is CPU and can run locally or in-kernel). Follow the Kaggle CLI workflow in the user's global `~/.claude/CLAUDE.md` (dataset deploy → kernel push → poll status → fetch output; handle the P100 sm_60 torch reinstall; `SSL_CERT_FILE` on macOS).
