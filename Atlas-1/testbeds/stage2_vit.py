"""Stage 2 — stretch: tiny ViT on colored shapes (§4)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Stage2Config:
    image_size: int = 64
    num_shapes: int = 4
    batch_size: int = 128
    epochs: int = 50


def build_dataset(config: Stage2Config):
    """Shape × hue (S^1) × rotation × scale rendered dataset."""
    _ = config
    raise NotImplementedError("Phase 4: rendered shape dataset")


def train_vit(config: Stage2Config):
    """Train tiny ViT classifier; return checkpoint path."""
    _ = config
    raise NotImplementedError("Phase 4: ViT training")
