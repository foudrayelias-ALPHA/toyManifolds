"""Hand-built point clouds for Phase 0 telemetry sanity checks."""

from __future__ import annotations

import numpy as np


def _embed_curve(
    params: np.ndarray,
    curve: np.ndarray,
    ambient_dim: int,
    rng: np.random.Generator,
    *,
    noise: float = 0.0,
    normalize: bool = True,
) -> np.ndarray:
    weights = rng.normal(size=(ambient_dim, curve.shape[1]))
    cloud = curve @ weights.T
    if noise > 0:
        cloud += noise * rng.normal(size=cloud.shape)
    if normalize:
        cloud /= np.linalg.norm(cloud, axis=1, keepdims=True).clip(min=1e-8)
    return cloud


def circle_cloud(
    n: int = 500,
    ambient_dim: int = 128,
    noise: float = 0.03,
    seed: int = 0,
    *,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Circle S^1 embedded in R^ambient_dim with stored angular coordinate."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    curve = np.column_stack([np.cos(theta), np.sin(theta)])
    curve += noise * rng.normal(size=curve.shape)
    cloud = _embed_curve(theta, curve, ambient_dim, rng, normalize=normalize)
    return cloud, theta


def arc_cloud(
    n: int = 400,
    ambient_dim: int = 128,
    noise: float = 0.03,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Open interval embedded as a semicircular arc (no persistent loop)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)
    curve = np.column_stack([np.cos(np.pi * t), np.sin(np.pi * t)])
    curve += noise * rng.normal(size=curve.shape)
    cloud = _embed_curve(t, curve, ambient_dim, rng)
    return cloud, t


def blob_cloud(
    n_clusters: int = 3,
    points_per_cluster: int = 80,
    ambient_dim: int = 128,
    cluster_std: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Discrete clusters (SAE-friendly blobs) with cluster labels."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, ambient_dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    clouds = []
    labels = []
    for idx, center in enumerate(centers):
        points = center + cluster_std * rng.normal(size=(points_per_cluster, ambient_dim))
        points /= np.linalg.norm(points, axis=1, keepdims=True).clip(min=1e-8)
        clouds.append(points)
        labels.append(np.full(points_per_cluster, idx, dtype=int))

    return np.vstack(clouds), np.concatenate(labels)


def torus_cloud(
    n: int = 800,
    ambient_dim: int = 128,
    noise: float = 0.03,
    seed: int = 0,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Torus S^1 x S^1 embedded in high dimensions."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 2 * np.pi, size=n)
    v = rng.uniform(0, 2 * np.pi, size=n)
    curve = np.column_stack([np.cos(u), np.sin(u), np.cos(v), np.sin(v)])
    curve += noise * rng.normal(size=curve.shape)
    cloud = _embed_curve(u, curve, ambient_dim, rng)
    return cloud, (u, v)
