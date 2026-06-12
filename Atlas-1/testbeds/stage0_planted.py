"""Stage 0 — planted ground-truth synthetic activations (§4)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class FamilyGeometry(str, Enum):
    DISCRETE = "discrete"
    CIRCLE = "circle"
    INTERVAL = "interval"
    TORUS = "torus"


@dataclass
class PlantedFamily:
    name: str
    geometry: FamilyGeometry
    occurrence_prob: float
    b0: int
    b1: int
    coordinates: np.ndarray


@dataclass
class Stage0Config:
    ambient_dim: int = 128
    snr_db: float = 10.0
    seed: int = 0
    num_samples: int = 10_000


def default_family_menu() -> list[dict]:
    """Planted menu from §4."""
    return [
        {"name": "discrete", "geometry": FamilyGeometry.DISCRETE, "count": 5, "prob": 0.2},
        {"name": "circle", "geometry": FamilyGeometry.CIRCLE, "count": 2, "prob": 0.25},
        {"name": "interval", "geometry": FamilyGeometry.INTERVAL, "count": 2, "prob": 0.25},
        {"name": "torus", "geometry": FamilyGeometry.TORUS, "count": 1, "prob": 0.2},
        {"name": "rare_circle", "geometry": FamilyGeometry.CIRCLE, "count": 1, "prob": 0.025},
    ]


def generate_stage0(config: Stage0Config) -> tuple[np.ndarray, list[PlantedFamily]]:
    """Generate synthetic activations with stored ground-truth coordinates."""
    rng = np.random.default_rng(config.seed)
    d = config.ambient_dim
    x = np.zeros((config.num_samples, d), dtype=np.float64)
    families: list[PlantedFamily] = []

    # Minimal placeholder generator — full geometry planting in Phase 1.
    for i, spec in enumerate(default_family_menu()):
        coords = rng.random(config.num_samples)
        mask = rng.random(config.num_samples) < spec["prob"]
        direction = rng.normal(size=d)
        direction /= np.linalg.norm(direction)
        x[mask] += direction * coords[mask, None]
        families.append(
            PlantedFamily(
                name=f"{spec['name']}_{i}",
                geometry=spec["geometry"],
                occurrence_prob=spec["prob"],
                b0=1 if spec["geometry"] == FamilyGeometry.DISCRETE else 1,
                b1=0 if spec["geometry"] == FamilyGeometry.DISCRETE else 1,
                coordinates=coords,
            )
        )

    noise_scale = 10 ** (-config.snr_db / 20)
    x += rng.normal(scale=noise_scale, size=x.shape)
    return x, families
