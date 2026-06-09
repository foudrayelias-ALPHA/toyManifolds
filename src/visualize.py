"""Manifold graph visualization via Isomap embedding."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.manifold import MDS

from .data import NUM_DAYS, NUM_MONTHS
from .extract import load_model_and_extract
from .graph import (
    build_neighborhood_graph,
    geodesic_distance_matrix,
    separable_baseline_activations,
)


def flatten_grid(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(NUM_MONTHS * NUM_DAYS, -1)


def isomap_embed(
    activations: np.ndarray,
    n_components: int = 3,
) -> np.ndarray:
    """
    Embed activation cloud in R^n preserving geodesic structure.

    Uses classical MDS on exact geodesic distances from the neighborhood
    graph — equivalent to Isomap with all n-1 neighbors (no KNN approximation).
    """
    adj, _ = build_neighborhood_graph(activations)
    dist = geodesic_distance_matrix(adj)
    mds = MDS(
        n_components=n_components,
        metric="precomputed",
        init="classical_mds",
        random_state=0,
        max_iter=500,
        n_init=1,
    )
    return mds.fit_transform(dist)


def plot_manifold_mesh(
    coords_3d: np.ndarray,
    title: str,
    out_path: Path,
    color_values: np.ndarray | None = None,
) -> None:
    """Draw wireframe mesh connecting conceptual neighbors in 3D."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if color_values is None:
        colors = np.arange(coords_3d.shape[0])
    else:
        colors = color_values

    ax.scatter(
        coords_3d[:, 0],
        coords_3d[:, 1],
        coords_3d[:, 2],
        c=colors,
        cmap="twilight",
        s=40,
        depthshade=True,
    )

    for m in range(NUM_MONTHS):
        for d in range(NUM_DAYS):
            i = m * NUM_DAYS + d
            for dm, dd in [
                ((m + 1) % NUM_MONTHS, d),
                (m, (d + 1) % NUM_DAYS),
            ]:
                j = dm * NUM_DAYS + dd
                seg = np.array([coords_3d[i], coords_3d[j]])
                ax.plot(
                    seg[:, 0],
                    seg[:, 1],
                    seg[:, 2],
                    color="gray",
                    alpha=0.35,
                    linewidth=0.6,
                )

    ax.set_title(title)
    ax.set_xlabel("Isomap 1")
    ax.set_ylabel("Isomap 2")
    ax.set_zlabel("Isomap 3")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def visualize_manifolds(
    activations: np.ndarray,
    out_dir: Path,
) -> dict[str, np.ndarray]:
    """Produce learned-manifold and separable-baseline comparison plots."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Color by combined phase for visual continuity on torus
    from .data import concept_angles

    colors = np.zeros(NUM_MONTHS * NUM_DAYS)
    for m in range(NUM_MONTHS):
        for d in range(NUM_DAYS):
            t1, t2 = concept_angles(m, d)
            colors[m * NUM_DAYS + d] = (t1 + t2) / (2 * np.pi)

    learned_3d = isomap_embed(activations)
    plot_manifold_mesh(
        learned_3d,
        "Learned feature manifold (Isomap)",
        out_dir / "manifold_learned.png",
        color_values=colors,
    )

    separable = separable_baseline_activations(activations)
    separable_3d = isomap_embed(separable)
    plot_manifold_mesh(
        separable_3d,
        "Separable baseline (flat grid)",
        out_dir / "manifold_separable_baseline.png",
        color_values=colors,
    )

    # Side-by-side 2D projection
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, coords, title in zip(
        axes,
        [learned_3d, separable_3d],
        ["Learned manifold", "Separable baseline"],
    ):
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=colors, cmap="twilight", s=30)
        for m in range(NUM_MONTHS):
            for d in range(NUM_DAYS):
                i = m * NUM_DAYS + d
                for dm, dd in [
                    ((m + 1) % NUM_MONTHS, d),
                    (m, (d + 1) % NUM_DAYS),
                ]:
                    j = dm * NUM_DAYS + dd
                    ax.plot(
                        [coords[i, 0], coords[j, 0]],
                        [coords[i, 1], coords[j, 1]],
                        color="gray",
                        alpha=0.3,
                        linewidth=0.5,
                    )
        ax.set_title(title)
        ax.set_aspect("equal")
    fig.colorbar(sc, ax=axes, label="(θ1 + θ2) / 2π")
    plt.tight_layout()
    plt.savefig(out_dir / "manifold_comparison_2d.png", dpi=150)
    plt.close()

    np.savez(
        out_dir / "embeddings.npz",
        learned_3d=learned_3d,
        separable_3d=separable_3d,
        activations=activations,
    )

    return {"learned_3d": learned_3d, "separable_3d": separable_3d}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()

    data = load_model_and_extract(args.checkpoint, layer=args.layer)
    visualize_manifolds(data["activations"], args.out_dir)
    print(f"Saved figures to {args.out_dir}")


if __name__ == "__main__":
    main()
