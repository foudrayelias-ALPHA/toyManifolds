#!/usr/bin/env python3
"""End-to-end pipeline: train → extract → visualize → analyze."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.analyze import analyze_manifold
from src.extract import extract_activations, load_model_and_extract
from src.train import train
from src.visualize import visualize_manifolds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demonstrate features as manifolds in a toy transformer"
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--layer", type=int, default=-1, help="Layer to analyze (-1 = last)")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    ckpt_path = out_dir / "model.pt"

    if not args.skip_train:
        print("=== Training ===")
        train(
            epochs=args.epochs,
            lr=args.lr,
            d_model=args.d_model,
            n_layers=args.n_layers,
            out_dir=out_dir,
        )

    print("=== Extracting activations ===")
    data = load_model_and_extract(ckpt_path, layer=args.layer)
    import numpy as np

    np.savez(
        out_dir / "activations.npz",
        activations=data["activations"],
        targets=data["targets"],
        layer=data["layer"],
    )
    print(f"  layer {data['layer']}, shape {data['activations'].shape}")

    print("=== Visualizing manifold ===")
    fig_dir = out_dir / "figures"
    visualize_manifolds(data["activations"], fig_dir)
    print(f"  saved to {fig_dir}")

    print("=== Geometric analysis ===")
    results = analyze_manifold(data["activations"])
    import json

    analysis_path = out_dir / "analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(results, f, indent=2)

    s = results["summary"]
    print(f"  geodesic intrinsic dim: {s['geodesic_intrinsic_dim']:.2f} (expected 2)")
    print(f"  MDS elbow dimension:    {s['mds_elbow_dimension']}")
    print(f"  participation ratio:      {s['pca_participation_ratio']:.2f}")
    print(f"  extrinsic dimension:      {s['extrinsic_dim']}")
    print(f"  torus topology:         {s['topology_consistent_with_torus']}")
    print(f"  saved to {analysis_path}")


if __name__ == "__main__":
    main()
