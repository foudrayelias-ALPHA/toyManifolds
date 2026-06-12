"""Evaluation metrics M1–M7 and dilution (§6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from atlas1.topology import TopologyReadout


@dataclass
class MetricReport:
    m1_family_count_error: float
    m2_topology_accuracy: float
    m3_coordinate_fidelity: float
    m4_dilution: dict[str, float]
    m5_fvu: float
    m6_stability: float | None = None
    m7_steering_accuracy: float | None = None
    extras: dict[str, Any] | None = None


def hungarian_match_families(
    predicted: list[TopologyReadout],
    ground_truth: dict[str, np.ndarray],
) -> list[tuple[int, str]]:
    """Match recovered families to planted factors. Stub for Phase 1."""
    _ = (predicted, ground_truth)
    raise NotImplementedError("Phase 1: Hungarian matching on coordinate fidelity")


def family_count_error(live_groups: int, planted_k: int, *, torus_ambiguity: bool = True) -> float:
    """M1: absolute error in live family count (±1 acceptable when torus_ambiguity)."""
    err = abs(live_groups - planted_k)
    if torus_ambiguity and err == 1:
        return 0.0
    return float(err)


def topology_accuracy(matched_pairs: list[tuple[TopologyReadout, dict]]) -> float:
    """M2: fraction of matched families with correct (b0, b1)."""
    if not matched_pairs:
        return 0.0
    correct = sum(
        1
        for pred, gt in matched_pairs
        if pred.b0 == gt["b0"] and pred.b1 == gt["b1"]
    )
    return correct / len(matched_pairs)


def coordinate_fidelity(pred_coord: np.ndarray, gt_coord: np.ndarray, *, circular: bool = False) -> float:
    """M3: R² (or circular R²) between recovered and ground-truth coordinates."""
    _ = circular
    if len(pred_coord) != len(gt_coord):
        raise ValueError("coordinate lengths must match")
    ss_res = np.sum((pred_coord - gt_coord) ** 2)
    ss_tot = np.sum((gt_coord - gt_coord.mean()) ** 2)
    if ss_tot == 0:
        return 1.0
    return float(1.0 - ss_res / ss_tot)


def dilution_per_factor(attribution: np.ndarray) -> float:
    """M4: perplexity of variance attribution across latents/families."""
    p = attribution / attribution.sum()
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def fraction_variance_unexplained(x: np.ndarray, x_hat: np.ndarray) -> float:
    """M5: FVU for reconstruction parity checks."""
    residual = x - x_hat
    return float(residual.pow(2).sum() / x.pow(2).sum()) if hasattr(x, "pow") else float(
        np.sum(residual**2) / np.sum(x**2)
    )
