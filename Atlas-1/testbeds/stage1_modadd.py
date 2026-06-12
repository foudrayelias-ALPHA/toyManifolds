"""Stage 1 — grokked modular-addition transformer (§4)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SiteName = Literal["embeddings", "resid_mid", "resid_post"]


@dataclass
class ModAddConfig:
    prime: int = 113
    d_model: int = 128
    n_heads: int = 4
    weight_decay: float = 1.0
    checkpoint_dir: Path = Path("runs/stage1/checkpoints")
    site: SiteName = "resid_post"


def train_and_grok(config: ModAddConfig):
    """Train Nanda-style 1-layer transformer until grokking; save checkpoint."""
    _ = config
    raise NotImplementedError("Phase 2: transformer_lens training pipeline")


def extract_activations(config: ModAddConfig, checkpoint: Path):
    """Featurize all p^2 inputs at the chosen site."""
    _ = (config, checkpoint)
    raise NotImplementedError("Phase 2: activation extraction over full input lattice")


def key_frequencies_from_model(checkpoint: Path) -> list[int]:
    """Identify model key frequencies k via FFT / logit attribution."""
    _ = checkpoint
    raise NotImplementedError("Phase 2: Fourier-circuit frequency identification")
