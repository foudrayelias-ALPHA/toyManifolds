"""Month × day-of-week concept space (12×7 torus) with non-separable targets."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

NUM_MONTHS = 12
NUM_DAYS = 7
VOCAB_SIZE = NUM_MONTHS * NUM_DAYS


def token_id(month: int, day: int) -> int:
    """Map (month, day) with month in [0,11], day in [0,6] to token id."""
    return month * NUM_DAYS + day


def decode_token(token_id: int) -> tuple[int, int]:
    month = token_id // NUM_DAYS
    day = token_id % NUM_DAYS
    return month, day


def month_day_grid() -> list[tuple[int, int]]:
    return [(m, d) for m in range(NUM_MONTHS) for d in range(NUM_DAYS)]


def concept_angles(month: int, day: int) -> tuple[float, float]:
    """Cyclic concept parameters θ1 (month), θ2 (day-of-week)."""
    theta1 = 2 * math.pi * month / NUM_MONTHS
    theta2 = 2 * math.pi * day / NUM_DAYS
    return theta1, theta2


def nonseparable_target(month: int, day: int) -> float:
    """
    Target that cannot be written as f(θ1) + g(θ2) or f(θ1) * g(θ2).

    Uses sin(θ1 + θ2) plus a mixed-frequency term sin(3θ1 + 2θ2).
    """
    theta1, theta2 = concept_angles(month, day)
    return math.sin(theta1 + theta2) + 0.5 * math.sin(3 * theta1 + 2 * theta2)


@dataclass
class ConceptSample:
    token_id: int
    month: int
    day: int
    target: float


def all_concept_samples() -> list[ConceptSample]:
    samples = []
    for month, day in month_day_grid():
        tid = token_id(month, day)
        samples.append(
            ConceptSample(
                token_id=tid,
                month=month,
                day=day,
                target=nonseparable_target(month, day),
            )
        )
    return samples


class MonthDayDataset(Dataset):
    """Single-token samples: predict non-separable scalar from (month, day) token."""

    def __init__(self) -> None:
        self.samples = all_concept_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            "token_id": torch.tensor(s.token_id, dtype=torch.long),
            "target": torch.tensor(s.target, dtype=torch.float32),
        }
