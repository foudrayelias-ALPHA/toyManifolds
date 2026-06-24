# Research Objective — Universal Manifolds Across Models

## Session

**Session:** `agent_logs/2026-06-21--universal-manifolds--lucid-glacier/`

---

## Objective

Determine whether a concept that two independently-trained language models both represent occupies
the **same intrinsic geometric structure** in each model — such that a low-distortion correspondence
between the two models' concept manifolds can be recovered **from geometry alone, without supervision**,
and whether that compute-discovered correspondence coincides with the human-interpretable concept.

---

## Motivation

If distinct models converge on the same internal geometry for a shared concept, that geometry is a
property of the *problem* (or of learning under shared data/architecture pressures), not an idiosyncrasy
of one network — the "universal/Platonic representation" question. The strong, falsifiable form of the
claim is not "a homeomorphism exists" (trivially true — all circles are homeomorphic) but "an
*unsupervised, low-distortion* map exists whose recovered correspondence matches human concepts without
ever being shown a label." That distinction is the whole point: we want to learn whether **compute on raw
geometry** rediscovers semantics we never imposed. Downstream, a positive result would make cross-model
transfer of interpretations, steering vectors, and probes a geometric exercise rather than a per-model
re-derivation; a negative result bounds how far representational universality actually extends.

---

## Scope boundaries

What is **not** investigated this session:

- **Out-of-scope models (this session):** gpt2 (cross-family contrast), llama31_70b, gemma4, base-vs-instruct
  comparisons. v1 is the llama 1B→8B scale ladder only. (All listed as later extensions.)
- **Out-of-scope concepts (v1 primary):** months, age, hours, graph_walk. `weekdays` is the primary domain;
  `alphabet` is included **only** as the discriminant control (different topology). `months` is named as a
  known cyclic-confound case but deferred.
- **Out-of-scope claim:** a single jointly-fit canonical/"Platonic" manifold with per-model distortion scores
  (a later extension). v1 does pairwise alignment only.
- **Out-of-scope mechanism:** *why* models converge (data overlap vs architectural prior vs scale). v1 measures
  *whether* they converge, not the cause.
- **Out-of-scope:** causal/steering validation that traversing the aligned manifold actually steers model B
  along model A's concept (that is a follow-up using `path_steering`/`pullback`, not v1).

---

## Success criteria

Concrete enough to write the conclusion paragraph today, numbers blank:

- **SC1 — Universality (metric).** For weekdays, the 1B↔8B manifold pair has distance-correlation
  (isometry-style Pearson, reusing `isometry.py`) **≥ 0.70**, and a Gromov-Wasserstein distance below the
  **5th percentile** of the shuffled-correspondence null (p < 0.05).
- **SC2 — Recovered semantics (unsupervised).** The label-free GW coupling recovers the weekday
  correspondence **≥ 6/7 correct up to the manifold's symmetry group** (global rotation/reflection of the
  7-cycle), with the raw (absolute-phase) accuracy reported alongside. *(See risk: a perfectly symmetric
  cycle admits no absolute phase — SC2 is stated modulo symmetry deliberately.)*
- **SC3 — Topology.** Persistent homology yields **b₁ = 1** (one loop) for both weekday manifolds and
  **b₁ = 0** for both alphabet manifolds; Betti numbers match within-concept across the two models.
- **SC4 — Discriminant validity.** Same-concept alignment cost (weekdays-1B↔weekdays-8B) is lower than
  cross-concept (weekdays-1B↔alphabet-8B) by a margin exceeding the shuffle-null spread.

If any are left unmet, that is itself a reportable bound on universality — not a failed session.

---

## Hypotheses

- **H1 — Near-isometry.** Independently-fit weekday manifolds in 1B and 8B are near-isometric (high
  distance-correlation, low GW distance) relative to shuffle/random nulls.
  *Falsified if* distance-correlation < 0.70 or GW distance is inside the shuffle-null bulk (p ≥ 0.05).
- **H2 — Unsupervised semantic recovery.** The no-label GW correspondence recovers the human weekday
  assignment well above chance (≥ 6/7 up to symmetry).
  *Falsified if* recovery is at chance (≈1/7) even after quotienting the symmetry group.
- **H3 — Topological match.** Betti numbers agree across models within a concept (weekdays b₁=1 both;
  alphabet b₁=0 both).
  *Falsified if* the two models' same-concept manifolds disagree on b₁.
- **H4 — Discriminant.** Same-concept cross-model alignment beats cross-concept alignment, modulo the
  cyclic-geometry confound (weekdays vs months would both be loops — measured later, not v1).
  *Falsified if* weekdays↔alphabet aligns as well as weekdays↔weekdays.
