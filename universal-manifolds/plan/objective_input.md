# Raw objective input (verbatim, for provenance)

User's free-form objective, captured at `/plan-experiment` Step 1.

---

I want to find 'universal manifolds', manifolds that persist across different models. The idea
is that this would find manifolds across models through the similar geometric structure. These
manifolds should ideally find concepts that are human-interpretable, but we should not be
intentionally trying to make them so. I want to leverage compute over human-imposed ideas. This
should be a very flexible framework. Maybe what we are looking for is a way to map a manifold
from model a to model b via a homeomorphism?

## Decisions reached during brainstorm

- **Method:** Gromov-Wasserstein optimal transport as the headline cross-model alignment (aligns
  metric spaces from intra-space distances only — no shared coordinates, no labels, handles
  different hidden dims). Procrustes orthogonal alignment as the linear baseline. Intrinsic-
  coordinate map (φ: U_A → U_B) as the interpretability lens.
- **Sharpened claim:** "there exists a homeomorphism" is too weak (all circles are homeomorphic).
  The real claim is a *low-distortion / near-isometric* map with a correspondence *recovered
  unsupervised* that nonetheless lands on human concepts (Monday↦Monday) without being told the
  labels. Homeomorphism is the floor; isometry + recovered semantics is the prize.
- **v1 models:** llama32_1b_instruct → llama31_8b (same-family scale ladder, highest probability
  of a clean first result). gpt2 cross-family contrast deferred.
- **v1 concept domain:** weekdays (clean cyclic geometry). alphabet used as the discriminant control.
- **Interpretability stance:** strictly post-hoc validation only — labels never enter fitting or
  the alignment objective; used only to *score* the recovered correspondence after the fact.
