# ATLAS-1: A Dictionary of Charts, Not Directions

**Mapping features in a transformer as manifolds instead of SAE latents — toy-scale, ground-truth-validated**

*Status: draft v0.1 · Companion to LOOP-1 (shared telemetry) · Scope: thorough — multi-seed, full ablation ladder · Compute: 1 GPU*

---

## 0. One-sentence version

Replace the SAE's dictionary of **directions** with a dictionary of **feature families**: an over-provisioned group-sparse autoencoder whose surviving groups are read out as manifolds (topology, intrinsic dimension, and a human-usable coordinate), validated first against planted ground truth and then against the provably-circular features of a grokked modular-addition transformer — with metrics as the gate and a coordinate-sweep demo as the deliverable.

The unit of interpretation becomes *(shape, coordinate, meaning)* instead of *(direction, label)*. A discrete SAE feature gets a name; an ATLAS family gets a name **and a knob**.

---

## 1. Prior work & positioning (read this before claiming novelty)

This idea is *in the air* as of 2025–2026. The diagnosis is published; the field has explicitly asked for the method; the validated constructive method appears open. State this plainly in any writeup.

| Work | What it did | Relation to ATLAS-1 |
|---|---|---|
| **Bhalla et al. 2026**, *Do Sparse Autoencoders Capture Concept Manifolds?* (arXiv:2604.28119) | Theory of SAE manifold capture: a **global** regime (compact atom group spanning the manifold) vs a **local** regime (atoms tiling patches); shows real SAEs land in a fragmented mix they call **dilution**. Calls for featurization methods that explicitly target manifolds and for group-level analysis tools. | The closest prior art. ATLAS-1 is a **constructive answer to their open call**: we *engineer* the global regime they show vanilla SAEs only partially reach. Their dilution concept becomes one of our headline metrics (§6). Position the writeup explicitly against this paper. |
| **Modell et al. 2025**, *The Origins of Representation Manifolds in LLMs* (arXiv:2505.18235) | Theory of features as metric spaces; proposes steering via the chart map $\phi_f: \mathcal{Z}_f \to \mathcal{M}_f$; conjectures feature splitting comes from SAE atoms tracing manifolds; hopes for "manifold-aware SAEs". | We implement the chart map they theorize, and use their steering proposal as our causal test (§6, M7). |
| **2025**, *Understanding SAE scaling in the presence of feature manifolds* (arXiv:2509.02565) | Shows SAEs reduce loss by tiling $S^1$ with more latents; analyzes pathological scaling. | Direct formalization of the "fractional recovery" thesis motivating this project. |
| **Engels et al. 2024**, *Not All Language Model Features Are Linear* (arXiv:2405.14860) | Found circular features (days, months, modular arithmetic) by clustering SAE decoder atoms; intervention evidence. | The empirical anchor. Their clustering pipeline is essentially our B2 baseline (SAE-then-stitch). |
| **Fel et al. 2025**, *Archetypal SAEs* (arXiv:2502.12892) | Constrains atoms to the data's convex hull for run-to-run stability. | Adjacent (stability, not manifold extraction). Motivates our stability metric M6 at the *family* level. |
| **Chen, Paiton & Olshausen 2018**, *The Sparse Manifold Transform* | Sparse coding + slow/manifold structure, pre-LLM. | Ancestor of the whole program; cite. |

**The gap ATLAS-1 fills:** an *unsupervised, end-to-end* featurizer whose first-class outputs are manifolds (not directions, not post-hoc clusters), with (i) recovery validated against **planted ground truth**, (ii) a **topology readout** (persistent homology, not just visual inspection), (iii) a **causal chart-steering test**, and (iv) head-to-head dilution comparison against vanilla SAEs and SAE-then-stitch.

> ⚠️ The literature is moving fast (Bhalla et al. appeared ~2 months ago). **Re-run a literature search immediately before any writeup**; if someone has built the constructive method by then, reposition as replication + the planted-ground-truth benchmark, which would still be a contribution.

---

## 2. Hypotheses & predictions ledger

| ID | Prediction | Prior | Status |
|----|-----------|-------|--------|
| **H1 — Planted recovery** | On Stage-0 synthetic data, ATLAS recovers the atlas: live-family count within ±1 of planted (modulo the pre-registered torus ambiguity), ≥90% of matched families with correct $(b_0, b_1)$, mean coordinate fidelity $R^2 \ge 0.9$. | 0.70 | — |
| **H2 — Real-model recovery** | On the grokked mod-add transformer, ≥1 recovered family per key frequency $k$ is a loop whose circular coordinate tracks $2\pi k(a+b)/p$ with circular $R^2 \ge 0.8$. | 0.55 | — |
| **H3 — Dilution beaten** | At matched latent budget and matched reconstruction, vanilla SAE shows dilution (effective latents-per-factor $\gg 1$ for manifold factors); ATLAS achieves $\approx 1$ family per factor. | 0.60 | — |
| **H4 — End-to-end earns its keep** | SAE-then-stitch (B2) closes only part of the gap to ATLAS on coordinate fidelity and dilution. *(Genuinely uncertain — a negative here is publishable: "tooling, not objectives.")* | 0.45 | — |
| **H5 — Families are stabler than atoms** | Across seeds, matched ATLAS family *subspaces* are more similar (principal-angle cosine) than matched individual SAE atoms. | 0.50 | — |
| **H6 — The chart steers** | Rotating a recovered sum-circle family by $\delta$ shifts the model's predicted answer by the chart-predicted amount $\delta \cdot p / (2\pi k) \bmod p$, ≥80% top-1 on intervened inputs. | 0.50 | — |

**Falsification:** H1 failing kills the method (the pipeline gate, §7). H2 failing while H1 passes means real features don't organize as cleanly as planted ones — report it; that is the experiment working. H3+H4 both failing means group objectives add nothing over existing SAE practice — also worth reporting.

---

## 3. Method: the Atlas featurizer

### 3.1 Objective (group-sparse autoencoder)

Latents partitioned into $J$ groups of $n$ (over-provisioned). Encoder $a = \mathrm{ReLU}(W_e x + b_e)$, unit-norm decoder atoms $D$, loss

$$\mathcal{L} = \big\| x - D a \big\|_2^2 \;+\; \lambda_g \sum_{j=1}^{J} \sqrt{n}\, \big\| a_{G_j} \big\|_2 \;+\; \lambda_{\text{in}} \big\| a \big\|_1 .$$

- **Group lasso** ($\ell_1$ over groups of $\ell_2$): whole groups activate sparsely; within-group co-activation is free → the cheapest encoding of a continuous family is one dedicated group. The only prior imposed: *features come in co-activating families of bounded size*. No shape, dimension, or topology is dictated.
- **Group death = family-count selection.** Over-provision $J$; unused groups are zeroed by the penalty. Report live-group count vs $\lambda_g$ (this curve is itself a finding).
- **Small within-group $\ell_1$** ($\lambda_{\text{in}} \ll \lambda_g$): guards against the *subspace-collapse* failure mode where a group uses its $n$ latents as a generic linear subspace rather than a curved family (sparse group lasso). Ablation A1 sweeps it; the intrinsic-dimension readout detects collapse either way.

### 3.2 Manifold readout (reuses LOOP-1 §5 telemetry verbatim)

For each live group $j$, over all inputs where $\|a_{G_j}\|_2 > \theta_{\text{act}}$:

1. **Family cloud** $\mathcal{C}_j = \{\, D_{G_j} a_{G_j}(x) \,/\, \|D_{G_j} a_{G_j}(x)\| \,\}$ — normalized group reconstructions.
2. **Topology**: Vietoris–Rips persistence (`ripser`, maxdim 1; DTM fallback) → loop score $\ell_j$, component count $\hat b_0$; TwoNN intrinsic dimension; usage.
3. **Coordinate**: if $\ell_j > \ell_{\text{crit}}$ (null-calibrated, as in LOOP-1) → circular coordinate from the dominant cocycle (`dreimac`); else if ID $\approx 1$ → 1-D coordinate (PCA/Isomap); else 2-D embedding.
4. **Chart map**: fit smooth $\hat g_j(\theta)$ (Fourier regression of cloud on circular coordinate; spline on linear) — this is the $\phi_f$ of Modell et al., and the object used for steering.
5. **Meaning**: coordinate sweep — bin the dataset by $\theta$, display exemplars per bin. *Feature visualization along the manifold.*

### 3.3 What is reported per family

$(\,\hat b_0,\ \hat b_1,\ \text{ID},\ \text{usage},\ \text{coordinate type},\ \text{coordinate fidelity vs matched ground truth},\ \text{sweep panel}\,)$ — one row per family, one panel per family. The atlas *is* the deliverable.

---

## 4. Testbeds

### Stage 0 — Planted ground truth (no network; method validation)

Synthetic activations $x = \sum_k s_k\, c_k\, g_k(z_k) + \varepsilon \in \mathbb{R}^{128}$ with a planted menu:

| Family | Count | Geometry | Notes |
|---|---|---|---|
| Discrete directions | 5 | points | the SAE-friendly case |
| Circles | 2 | $S^1$, random 2-D+curved embeddings, different radii | the headline case |
| Intervals | 2 | $[0,1]$ arcs | open families (boundary) |
| Torus | 1 | $S^1 \times S^1$ | **pre-registered ambiguity**: recovering it as one 2-D family *or* two circular families both count as correct |
| Rare circle | 1 | $S^1$, occurrence prob 10× lower | density-warping stress test |

Sparse occurrence $s_k \sim \text{Bern}(p_k)$, amplitudes $c_k$ log-normal, noise $\varepsilon$ at three SNR levels. All $z_k$ stored — every metric is exact.

### Stage 1 — Grokked modular-addition transformer (primary)

- Model: 1-layer transformer (Nanda-style), $d_{\text{model}} = 128$, 4 heads, sequence $[a, b, =]$, $p = 113$, trained past grokking with weight decay 1.0, then **frozen**. (`transformer_lens` recommended.)
- Featurize: `resid_post` at the final position over all $p^2 = 12{,}769$ inputs (exact statistics, zero sampling error). Secondary sites as ablation A6: embeddings (token circles per frequency), `resid_mid`.
- Ground truth: the Fourier-circuit story — identify the model's key frequencies $\{k\}$ (FFT of embedding columns / logit attribution), giving target coordinates $2\pi k(a+b)/p$. Matching = best circular–circular correlation over $k$.
- **Steering test (H6/M7):** for a recovered sum-circle family with chart $\hat g_j$, patch $x' = x - \hat x_j(\theta) + \hat x_j(\theta + \delta)$ and check the predicted answer shifts by $\delta\, p/(2\pi k) \bmod p$. Choose $\delta = 2\pi k\, m / p$ for integer $m$ so the predicted shift is exactly $m$ residues; sweep $m$ over the circle; report top-1 shift accuracy.

### Stage 2 — Stretch: tiny ViT on colored shapes (the human-legibility demo)

Rendered dataset: shape identity (categorical) × hue ($S^1$ — the most human-readable circle in existence) × rotation × scale. Tiny ViT classifier, featurize a mid-block residual. Two sharp predictions: the hue family is a loop whose sweep panel is literally a color wheel; and a **square's rotation family is the quotient** $S^1/C_4$ — persistence sees a loop either way, but the coordinate regression must show frequency 4. Symmetry quotients become falsifiable predictions instead of confounds. *(Alternative stretch if vision tooling is unwanted: tiny LM on synthetic token sequences with planted cyclic/ordinal generative factors.)*

---

## 5. Baselines, controls, ablations (the thorough ladder)

| ID | Run | Tests |
|----|-----|-------|
| **B1** | Vanilla SAE, matched total latents $J{\cdot}n$, matched reconstruction | The incumbent; source of the dilution measurement |
| **B2** | SAE-then-stitch: cluster B1 decoder atoms (co-activation + cosine graph, Leiden), identical readout per cluster | The Engels-style pipeline; H4 |
| **B3** | Oracle stitch (Stage 1 only): atoms grouped using *ground-truth labels* | Upper bound on any post-hoc grouping |
| **C1** | Shuffled activations (destroy structure) | Pipeline null: no loops, no coordinates |
| **C2** | Random-init featurizer | Null distribution for $\ell_{\text{crit}}$ (as in LOOP-1) |
| **A1** | $\lambda_{\text{in}} \in \{0, 10^{-4}, 10^{-3}\}$ | Subspace-collapse failure mode |
| **A2** | Group size $n \in \{4, 16, 64\}$ | Capacity per family |
| **A3** | Group count $J$ at 0.5×, 1×, 4× planted family count | Over/under-provisioning robustness |
| **A4** | Group-TopK activation vs group lasso | Objective form |
| **A5** | Readout on raw group decoder atoms vs reconstruction clouds | Cheaper readout viable? |
| **A6** | Featurization site (embeddings / resid_mid / resid_post) | Where families live |

All headline runs: **5 seeds**. Stage-0 metric tables report mean ± std across seeds.

---

## 6. Metrics (the gate)

- **M1 Family count**: live groups vs planted $K$ (±1, torus ambiguity pre-registered).
- **M2 Topology accuracy**: fraction of matched families with correct $(b_0, b_1)$.
- **M3 Coordinate fidelity**: circular $R^2$ (cyclic) / $R^2$ (linear) vs ground-truth $z$, after Hungarian matching of families to factors.
- **M4 Dilution (operationalizing Bhalla et al.)**: for each ground-truth factor, attribute its variance across latents (B1) or families (ATLAS); **effective units per factor** = perplexity of that attribution distribution. Target: ATLAS ≈ 1 per manifold factor; B1 $\gg 1$ for manifold factors and ≈ 1 for categorical ones (the contrast is itself a signature).
- **M5 Reconstruction parity**: FVU within $\epsilon = 0.02$ of B1 — coherence must not be purchased with reconstruction.
- **M6 Stability**: across seeds, mean principal-angle cosine between matched family subspaces (ATLAS) vs matched atom cosine (B1).
- **M7 Steering accuracy** (Stage 1): chart-predicted answer-shift top-1 rate.

**Gates** (balanced mandate: metrics gate, demo deliverable):
- *Stage 0 → Stage 1*: M1 within ±1, M2 ≥ 0.9, M3 ≥ 0.9 at mid SNR.
- *Stage 1 → demo polish*: H2 satisfied for ≥1 frequency **and** M7 ≥ 0.8.

---

## 7. Phases

1. **Phase 0 — Telemetry sanity** (shared with LOOP-1 Phase 0): persistence + coordinate pipeline verified on hand-built circle/blobs/arc clouds. *Hard gate.*
2. **Phase 1 — Stage 0 recovery**: full ladder on planted data, 5 seeds, three SNRs. *Gate: §6.*
3. **Phase 2 — Stage 1 mod-add**: train/grok/freeze the transformer; featurize; H2–H6; steering.
4. **Phase 3 — Demo**: the **knob demo** — an interactive page/notebook: pick a family → see its 3-D cloud colored by coordinate (deliberately in the format of the SAE-feature-manifold scatter panels: open curve / loop / clusters) → drag a slider along the coordinate → watch exemplars and (Stage 1) predicted-logit shift update live.
5. **Phase 4 — Stretch**: Stage 2 ViT, the color wheel and the $S^1/C_4$ prediction.
6. **Phase 5 — Writeup**: fresh literature pass; position vs Bhalla et al. (we engineer their global regime), Modell et al. (we implement their chart map and steering proposal).

---

## 8. Deliverables

- **F1**: Stage-0 recovery scorecard (planted vs recovered, all metrics, all seeds).
- **F2**: **The atlas panel** — per-family 3-D PCA clouds colored by recovered coordinate.
- **F3**: $(a,b)$-lattice heatmaps colored by each family's coordinate — sum-circles appear as diagonal stripes at frequency $k$.
- **F4**: Dilution bar chart — effective units per factor: ATLAS vs B1 vs B2 vs B3.
- **F5**: Steering plot — intended vs realized answer shift across $\delta$.
- **F6**: Stability curves across seeds (family subspaces vs atoms).
- **D1**: The knob demo (Phase 3).
- Repo: `featurizer.py`, `topology.py` (shared with LOOP-1), `testbeds/`, `metrics.py`, `demo/`; every figure regenerable from saved runs.

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Subspace collapse (families = flat subspaces) | $\lambda_{\text{in}}$ knob (A1); ID-vs-group-rank readout reports it honestly either way |
| One factor split across groups / two factors merged | Hungarian matching exposes it; A3 sweep; report sensitivity to $J$ |
| Torus ambiguity (one 2-D family vs two circles) | Pre-registered: both accepted (identifiability is not solvable by resolution) |
| Rare-family loss (density warping) | Stage-0 rare circle is the canary; report recovery vs occurrence rate |
| All groups die / none die | $\lambda_g$ sweep with live-count curve as a standard diagnostic |
| Mod-add features less tidy than planted ones | That is a *finding*, not a failure — the planted/real gap is the point of having both stages |
| Scooped mid-project | §1 warning: fresh search pre-writeup; benchmark + ground-truth suite remains contributable |

---

## 10. Defaults

| | | | |
|---|---|---|---|
| ambient dim $d$ | 128 | groups $J$ | 64 (Stage 0), 32 (Stage 1) |
| group size $n$ | 16 | total latents | 1024 / 512 |
| $\lambda_g$ | sweep {1e-3…1e-1}, pick by live-count elbow | $\lambda_{\text{in}}$ | 1e-4 |
| $\theta_{\text{act}}$ | 0.1 × max group norm | $\ell_{\text{crit}}$ | 99th pct of C2 nulls |
| optimizer | Adam 1e-3 | steps | 50k (featurizer) |
| seeds | 5 | $p$ | 113 |
| FPS cap | 1,000 pts | PCA dims (pre-Rips) | 10 |

**Dependencies**: `torch`, `transformer_lens` (Stage 1 model), `ripser`, `persim`, `dreimac`, `giotto-tda`, `scikit-dimension`. Compute: every featurizer run is seconds-to-minutes (≤13k activation vectors, $d = 128$); the full ladder is hours, not days.

## 11. Relationship to LOOP-1

Same worldview, two entry points. **ATLAS-1** (this doc): manifolds as the *output of a decomposition method* on a frozen standard transformer — do this first; no architecture risk. **LOOP-1**: manifolds as *emergent objects inside a redesigned architecture* — the sequel, with shared `topology.py` telemetry. A side quest compatible with both: the splitting-rate scaling law (nearest-atom spacing vs dictionary size $\sim J^{-1/m}$) measured on public SAE suites trained at multiple widths — pure analysis, no training, and a direct observational test of the "SAEs quantize manifolds" thesis.
