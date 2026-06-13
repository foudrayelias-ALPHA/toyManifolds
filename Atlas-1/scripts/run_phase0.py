#!/usr/bin/env python3
"""Phase 0 — verify persistence + coordinate pipeline on hand-built clouds."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from atlas1.synthetic_clouds import arc_cloud, blob_cloud, circle_cloud
from atlas1.topology import CoordinateType, ReadoutConfig, readout_family


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def _check_circle(config: ReadoutConfig) -> CheckResult:
    cloud, theta = circle_cloud(seed=0)
    readout = readout_family(0, cloud, config, ground_truth=theta)
    ok = (
        readout.b1 >= 1
        and readout.loop_score > config.loop_crit
        and readout.coordinate_type == CoordinateType.CIRCULAR
        and (readout.coordinate_fidelity or 0.0) >= 0.8
    )
    return CheckResult(
        "circle",
        ok,
        f"b1={readout.b1}, loop={readout.loop_score:.3f}, "
        f"type={readout.coordinate_type.value}, R2={readout.coordinate_fidelity:.3f}, "
        f"ID={readout.intrinsic_dim:.2f}",
    )


def _check_arc(config: ReadoutConfig) -> CheckResult:
    cloud, t = arc_cloud(seed=1)
    readout = readout_family(1, cloud, config, ground_truth=t)
    ok = readout.loop_score < config.loop_crit and readout.coordinate_type in {
        CoordinateType.LINEAR,
        CoordinateType.EMBEDDING_2D,
    }
    return CheckResult(
        "arc",
        ok,
        f"b1={readout.b1}, loop={readout.loop_score:.3f}, "
        f"type={readout.coordinate_type.value}, ID={readout.intrinsic_dim:.2f}",
    )


def _check_blobs(config: ReadoutConfig) -> CheckResult:
    cloud, labels = blob_cloud(n_clusters=3, seed=2)
    readout = readout_family(2, cloud, config)
    ok = readout.b0 >= 3 and readout.b0 <= 5 and readout.loop_score < config.loop_crit
    return CheckResult(
        "blobs",
        ok,
        f"b0={readout.b0}, b1={readout.b1}, loop={readout.loop_score:.3f}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ATLAS-1 Phase 0 telemetry sanity")
    parser.add_argument("--loop-crit", type=float, default=0.3)
    args = parser.parse_args()

    config = ReadoutConfig(loop_crit=args.loop_crit)
    checks = [_check_circle(config), _check_arc(config), _check_blobs(config)]

    print("=== ATLAS-1 Phase 0: Telemetry Sanity ===")
    passed = 0
    for result in checks:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")
        passed += int(result.passed)

    print(f"\n{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
