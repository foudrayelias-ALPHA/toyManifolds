# Continuity & provenance

This folder is a self-contained snapshot of a research investigation. It is here so the work can be picked up
later (by a person or an agent) without re-deriving context. The live working copy is a causalab research session.

## Origin

- **Framework**: built as a *session-local extension* to [causalab](https://github.com/goodfire-ai/causalab)
  (mechanistic interpretability via causal abstraction). causalab supplied the task definitions
  (`natural_domains_arithmetic`), activation extraction, PCA subspace fitting (`subspace`), and the Hydra runner.
  The `code/` here is the new contribution that plugs into it.
- **Live session**: `causalab/agent_logs/2026-06-21--universal-manifolds--lucid-glacier/` (on the machine it was
  built on). That session holds the full 636 MB artifact tree (baseline output-dists, manifold checkpoints) not
  vendored here. This folder keeps only the small inputs/outputs needed to read and reproduce the result.
- **How it was built**: interactively, June 2026 — started as "explain this repo", became a full
  plan → custom-methods → producers → cross-model-alignment → report cycle.

## The arc (what happened, in order)

1. **Objective** crystallized: are concept manifolds *universal* across independently-trained models, recoverable
   unsupervised? (`plan/RESEARCH_OBJECTIVE.md`, `plan/PLAN.md`.)
2. **Custom code** scaffolded + unit-tested: `cross_model_alignment` (GW/Procrustes/label-recovery/nulls),
   `manifold_topology` (PH/Betti), and the `universal_manifold` analysis.
3. **Producers** ran: GPT-2 + Llama-3.2-1B locally; **Llama-3.1-8B on a Kaggle T4 GPU** (headless kernel).
4. **Aligners**: 12 scale (1B↔8B) + 12 cross-family (gpt2↔8B) alignments at matched relative depths.
5. **Report** written and then extended with the gpt2 cross-family result (`REPORT.md`).

## Key decisions (and why)

- **Target the `entity` token, not `result`.** The day-of-week *input* concept is cleanly encoded at any
  competence; the computed *answer* is unreliable (1B solves weekdays only 18% of the time, GPT-2 0%). Fitting on
  `entity` is what makes the cross-competence comparison meaningful.
- **Depth sweep on the concept token, not locate's "best cell".** Localization picks the *answer-formation* site
  (last token, late layer); the clean concept manifold lives at the *entity* token. We sweep matched relative
  depths {0, .25, .5, .75} so 1B/8B/gpt2 are comparable.
- **Gromov-Wasserstein as the headline aligner.** Label-free, dimension-agnostic (gpt2 768-d ↔ Llama 4096-d, both
  PCA-8), returns both a correspondence and a distance. Labels enter *only* the post-hoc recovery score.

## Gotchas found + fixed (see `issues.md`)

- **GW was not scale-invariant** → deep layers (large activation norms) underflowed to spurious GW=0. Fixed by
  normalizing each distance matrix to unit max. (All reported numbers are post-fix.)
- **Topology by Betti is unreliable here.** Vietoris-Rips PH on 7-point/8-D centroids gives a vanishing H1 bar
  (the loop fills instantly in high-D); 2-NN-graph Betti over-counts cycles. The **ring-adjacency (cyclic-NN)
  fraction** is the reliable structural metric (1.00 weekdays vs 0.35 alphabet). Promote `ring_adjacency` to a
  first-class method when this graduates.
- **Kaggle**: first 8B kernel failed because Hydra tried to log into read-only `/kaggle/input`; fixed with
  `cwd=/kaggle/working`. Use ungated HF mirrors (`unsloth/Llama-3.2-1B-Instruct`, `unsloth/Meta-Llama-3.1-8B`) +
  `enable_internet` to sidestep gating; guard for P100 (sm_60) torch.

## Compute provenance

| model | role | where | task accuracy (weekdays) |
|---|---|---|---|
| GPT-2 124M (`gpt2`) | cross-family | local CPU/MPS | 0% |
| Llama-3.2-1B-Instruct (`unsloth` mirror) | small scale | local CPU/MPS | 18% |
| Llama-3.1-8B (`unsloth/Meta-Llama-3.1-8B`) | large scale | Kaggle T4 kernel `causalab-um-8b` | 94% |

## Status & next steps (priority order)

1. **`months`** (12-cycle): does universality track the *specific* concept or generic cyclic structure?
2. **Computed-`result` manifold on the 8B** (94% accurate): does the *computation's* geometry universalize too?
3. **Joint canonical manifold**: fit one shared "platonic" weekday manifold, score each model's distortion to it.
4. **Scale to 70B**; add more concept families to map where cross-architecture universality holds/breaks.
5. **Promote** `cross_model_alignment`, `manifold_topology` (+ a new `ring_adjacency`), and `universal_manifold`
   into the causalab tree; they are stable and tested.

## How to continue

- Read `REPORT.md` (the verdict) and `plan/PLAN.md` (the design).
- The aligner reads inputs from `data/manifolds/<model>/<concept>/` (feature clouds + `train_dataset.json` labels);
  every result is in `data/alignments/<pair>/metrics.json`.
- To re-run end-to-end, restore the causalab framework and the session configs under `code/configs/runners/`.
