#!/usr/bin/env python3
"""Run the full baseline/ablation ladder (§5)."""

from __future__ import annotations

import argparse

from atlas1.baselines import ABLATIONS, ladder_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="ATLAS-1 experiment ladder")
    parser.add_argument("--stage", choices=["0", "1"], default="0")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--list-ablations", action="store_true")
    args = parser.parse_args()

    if args.list_ablations:
        for spec in ABLATIONS.values():
            print(f"{spec.run_id}: {spec.description} → {spec.sweep}")
        return

    runs = ladder_runs(include_stretch=args.stage == "1")
    print(f"Stage {args.stage} ladder ({args.seeds} seeds): {[r.value for r in runs]}")
    print("Orchestration wiring is a Phase 1+ task.")


if __name__ == "__main__":
    main()
