"""Extract residual-stream activations for every concept combination."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .data import NUM_DAYS, NUM_MONTHS, all_concept_samples
from .model import ToyTransformer


def extract_activations(
    model: ToyTransformer,
    layer: int = -1,
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    """
    Collect activations for all 12×7 concept combinations.

    Returns:
        activations: (12, 7, d_model) array
        targets: (12, 7) non-separable target values
        token_ids: (12, 7) token indices
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    samples = all_concept_samples()
    d_model = model.d_model

    activations = np.zeros((NUM_MONTHS, NUM_DAYS, d_model), dtype=np.float32)
    targets = np.zeros((NUM_MONTHS, NUM_DAYS), dtype=np.float32)
    token_grid = np.zeros((NUM_MONTHS, NUM_DAYS), dtype=np.int64)

    with torch.no_grad():
        for s in samples:
            tid = torch.tensor([s.token_id], device=device)
            _, layer_acts = model(tid, return_activations=True)
            act = layer_acts[layer].cpu().numpy()[0]
            activations[s.month, s.day] = act
            targets[s.month, s.day] = s.target
            token_grid[s.month, s.day] = s.token_id

    return {
        "activations": activations,
        "targets": targets,
        "token_ids": token_grid,
        "layer": layer if layer >= 0 else model.n_layers + layer,
    }


def load_model_and_extract(
    checkpoint_path: Path,
    layer: int = -1,
) -> dict[str, np.ndarray]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = ToyTransformer(
        d_model=config["d_model"],
        n_layers=config["n_layers"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return extract_activations(model, layer=layer)
