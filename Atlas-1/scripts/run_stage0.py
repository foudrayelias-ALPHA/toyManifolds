#!/usr/bin/env python3
"""Run Stage 0 planted-ground-truth recovery (§7 Phase 1)."""

from __future__ import annotations

import argparse
from pathlib import Path

from testbeds.stage0_planted import Stage0Config, generate_stage0


def main() -> None:
    parser = argparse.ArgumentParser(description="ATLAS-1 Stage 0 recovery")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/stage0"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snr-db", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true", help="Generate data only")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = Stage0Config(seed=args.seed, snr_db=args.snr_db)
    activations, families = generate_stage0(config)

    import numpy as np

    np.save(args.out_dir / "activations.npy", activations)
    print(f"Wrote {activations.shape} activations to {args.out_dir}")

    if args.dry_run:
        print(f"Planted families: {len(families)}")
        return

    print("Full ladder not yet implemented — structure ready for Phase 1.")


if __name__ == "__main__":
    main()
