#!/usr/bin/env python3
"""Run Stage 1 grokked mod-add pipeline (§7 Phase 2)."""

from __future__ import annotations

import argparse
from pathlib import Path

from testbeds.stage1_modadd import ModAddConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="ATLAS-1 Stage 1 mod-add")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/stage1"))
    parser.add_argument("--site", default="resid_post")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = ModAddConfig(checkpoint_dir=args.out_dir / "checkpoints", site=args.site)
    print(f"Stage 1 scaffold ready (site={config.site}). Implement in Phase 2.")
    print(f"Outputs will go to {args.out_dir}")


if __name__ == "__main__":
    main()
