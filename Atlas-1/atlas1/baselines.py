"""Baselines B1–B3, controls C1–C2, and ablation registry (§5)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable


class RunKind(str, Enum):
    ATLAS = "atlas"
    B1_VANILLA_SAE = "b1_vanilla_sae"
    B2_SAE_STITCH = "b2_sae_stitch"
    B3_ORACLE_STITCH = "b3_oracle_stitch"
    C1_SHUFFLED = "c1_shuffled"
    C2_RANDOM_INIT = "c2_random_init"


@dataclass(frozen=True)
class AblationSpec:
    run_id: str
    description: str
    sweep: dict[str, list]


ABLATIONS: dict[str, AblationSpec] = {
    "A1": AblationSpec("A1", "Within-group L1 (subspace collapse)", {"lambda_in": [0.0, 1e-4, 1e-3]}),
    "A2": AblationSpec("A2", "Group size n", {"group_size": [4, 16, 64]}),
    "A3": AblationSpec("A3", "Group count J vs planted K", {"num_groups_scale": [0.5, 1.0, 4.0]}),
    "A4": AblationSpec("A4", "Group-TopK vs group lasso", {"objective": ["group_lasso", "group_topk"]}),
    "A5": AblationSpec("A5", "Readout on atoms vs clouds", {"readout_source": ["atoms", "clouds"]}),
    "A6": AblationSpec("A6", "Featurization site", {"site": ["embeddings", "resid_mid", "resid_post"]}),
}


def ladder_runs(*, include_stretch: bool = False) -> list[RunKind]:
    """Default thorough ladder (5 seeds each in full experiments)."""
    runs = [
        RunKind.ATLAS,
        RunKind.B1_VANILLA_SAE,
        RunKind.B2_SAE_STITCH,
        RunKind.C1_SHUFFLED,
        RunKind.C2_RANDOM_INIT,
    ]
    if include_stretch:
        runs.append(RunKind.B3_ORACLE_STITCH)
    return runs


def apply_control(kind: RunKind, activations, rng=None):
    """Return modified activations for control runs."""
    import numpy as np

    if kind == RunKind.C1_SHUFFLED:
        rng = np.random.default_rng(rng)
        flat = activations.reshape(len(activations), -1)
        idx = rng.permutation(len(flat))
        return flat[idx]
    return activations


# Placeholder hooks — implemented in later phases.
TrainFn = Callable[..., object]
ReadoutFn = Callable[..., object]
