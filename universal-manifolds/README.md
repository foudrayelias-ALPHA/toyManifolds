# Universal Manifolds — cross-model geometry of the weekday concept

**Do two independently-trained language models represent the same concept with the same intrinsic geometry —
recoverable from geometry alone, without supervision?** For the day-of-week concept, the answer is **yes, and
across architecture, not just scale.**

Three models spanning a 65× size range and two architecture families — **GPT-2 (124M)**, **Llama-3.2-1B**,
**Llama-3.1-8B** — each encode the seven weekdays as a clean **7-cycle**, and a label-free
[Gromov-Wasserstein](https://pythonot.github.io/) alignment between any two of these manifolds is a low-distortion
near-isometry that is **specific to the concept** (it does not hold against a different concept).

## Headline result

| weekday metric | across **scale** (1B↔8B) | across **architecture** (gpt2↔8B) |
|---|---|---|
| distance-correlation (isometry) | 0.83–0.91 | **0.78–0.89** |
| GW discriminant ratio (vs. a *different* concept, alphabet) | 12–39× | **7–20×** |
| unsupervised label recovery @ embedding layer | 5/7 | **7/7** |

The sharpest single fact: **GPT-2 scores 0/49 on weekday arithmetic — it cannot use the concept at all — yet it
still represents the days as a clean cyclic ring** (nearest-neighbour ring-adjacency = 1.00 at every depth), and
the unsupervised correspondence between GPT-2's and Llama-8B's day-rings recovers `Monday↦Monday` **perfectly** at
the embedding layer. The geometry is a property of the concept, not of any one network's competence.

![the weekday ring across depths in Llama-3.2-1B](figures/weekdays_1b_depth_sweep.png)

*Seven day-centroids connected in calendar order, at four depths in Llama-3.2-1B. The calendar order traces a
clean, non-crossing ring at every layer — and the same ring appears in GPT-2 and Llama-8B.*

See **[REPORT.md](REPORT.md)** for the full write-up (success criteria, hypotheses, every number cited to its data file).

## Method, in one paragraph

For a behavioural task (e.g. *"what day is N days after {day}?"*) we extract residual-stream activations at the
**concept token** (the input day), reduce to an 8-D PCA subspace per model/layer, and take per-class centroids.
Two models' concept manifolds are then aligned with **Gromov-Wasserstein optimal transport**, which matches metric
spaces using only *intra-space* distances — no shared coordinates, no labels, and it handles different hidden
dimensions natively. Universality is scored by (a) a scale-invariant distance-correlation, (b) the GW distance vs.
geometry-destroying nulls, (c) a same-concept-vs-different-concept discriminant, and (d) post-hoc, whether the
label-free correspondence lands on the human concept. Labels are used **only** to score the recovered map, never to
fit it — *compute over human-imposed ideas.*

## What's here

```
universal-manifolds/
├── README.md            this file
├── REPORT.md            full experiment report (verdict against SC1–SC4 / H1–H4)
├── CONTINUITY.md        provenance, key decisions, gotchas, and how to continue
├── issues.md            engineering log (bugs found + fixed during the run)
├── figures/             the headline figures
├── plan/                research objective + detailed plan + method/analysis specs
├── code/                the reusable contribution (see below)
│   ├── methods/
│   │   ├── cross_model_alignment/   GW + Procrustes + label-recovery + nulls
│   │   └── manifold_topology/       persistent homology + k-NN-graph Betti
│   ├── analyses/universal_manifold/ the cross-model aligner (Hydra entry point)
│   └── configs/                     analysis + runner configs
└── data/
    ├── alignments/      every alignment's metrics.json (the numbers behind the tables)
    └── manifolds/       the input PCA feature clouds + labels, per model/concept/layer
```

## Reproducing / extending

The **methods** (`code/methods/`) are near-standalone — they depend only on `numpy`, `POT` (`ot`), `scipy`, and
`ripser` — and the unit tests under each `tests/` run on synthetic data with no models. You can drive
`cross_model_alignment.gromov_wasserstein` directly on the cached clouds in `data/manifolds/`.

The **analysis and producers** were built as a session-local extension to the
[causalab](https://github.com/goodfire-ai/causalab) mechanistic-interpretability framework (it supplies the task
definitions, activation extraction, PCA subspace fitting, and the Hydra runner). To re-run the full pipeline you
need that framework; `CONTINUITY.md` documents the exact configs, models (ungated HF mirrors), and the Kaggle GPU
kernel used for the 8B model.

## Honest caveats

- The GW shuffle-null is permissive for 7-point centroids; the **discriminant** and **distance-correlation** are
  the load-bearing metrics.
- Absolute (not just up-to-symmetry) semantic recovery is perfect only at the embedding layer; ~5/7 deeper.
- Formal topological invariants (persistent homology, graph Betti) proved **unreliable** at N=7 / dim=8 — the
  nearest-neighbour ring-adjacency fraction (1.00 weekdays vs. 0.35 alphabet) is the reliable structural signal.
  This is itself a finding (see REPORT §9).

## License

Inherits the repository's MIT license. Built on [causalab](https://github.com/goodfire-ai/causalab) (the framework
this extends is not vendored here).
