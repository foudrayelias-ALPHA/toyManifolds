"""universal_manifold: is one model's concept manifold a low-distortion, unsupervised image of another's?

Cross-model analysis. Reads TWO producer experiment_roots (source + target). For each model it
loads the cached PCA-subspace feature cloud, groups it into per-class centroids, and then aligns the
two manifolds with Gromov-Wasserstein OT (label-free), scoring near-isometry (distance-correlation,
reusing the shipped isometry scorer), the Procrustes baseline, unsupervised label-recovery (raw +
up-to-cyclic-symmetry), topological invariants (Betti numbers), and geometry-destroying nulls.

Session-local *analysis* — see ARCHITECTURE.md §3:
  - depends on causalab/{io,runner.helpers,methods.scores}, never on causalab/analyses/ peers
  - all disk I/O via causalab.io.* primitives (invariant 3)
  - no hyperparameter defaults inline — every knob comes from cfg.universal_manifold.* / cfg.task.*
  - cfg.experiment_root is the single source of truth for output paths (invariant 7)
  - NO model weights loaded — operates purely on cached features (CPU).

First-run note: the per-example label loader (`_load_side`) assumes the subspace `features` rows are
in regenerated train-dataset order. If the producer filtered to correct-only examples the lengths
differ; the loader raises a clear error rather than misaligning. Validate on the first real run.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from causalab.io.artifacts import (
    save_experiment_metadata,
    save_json_results,
    save_tensor_results,
    load_tensor_results,
)
from causalab.methods.scores.isometry import compute_isometry_metrics

# session-local methods (resolved via CAUSALAB_SESSION_CODE PYTHONPATH injection)
from methods.cross_model_alignment import (
    coupling_to_matching,
    gromov_wasserstein,
    label_recovery,
    null_distribution,
    procrustes_align,
)
from methods.manifold_topology import betti_match, graph_betti, persistent_homology

logger = logging.getLogger(__name__)

ANALYSIS_NAME = "universal_manifold"


# --------------------------------------------------------------------------- loaders
def _subspace_dir(side: DictConfig, target_variable: str) -> str:
    """Resolve the subspace/<sub>/<target_variable>/ artifact dir for one side.

    Real layout (confirmed against producer output): the subspace subdir is keyed by the
    target variable, not by a layer_x_pos cell — e.g. subspace/pca_k8/entity/.
    """
    return os.path.join(
        side.experiment_root, side.variant, "subspace", side.subspace, target_variable
    )


def _features_dir(side: DictConfig, target_variable: str) -> str:
    """Feature dir for one side. Depth-sweep producers nest per-layer cells under
    ``<tv>/layer_x_pos/L{layer}_{pos}/features/``; a single-cell producer writes ``<tv>/features/``.
    Set side.layer (+ optional side.pos, default = target_variable) to pick the sweep cell.
    """
    base = _subspace_dir(side, target_variable)
    if side.get("layer") is not None:
        pos = side.get("pos") or target_variable
        return os.path.join(base, "layer_x_pos", f"L{side.layer}_{pos}", "features")
    return os.path.join(base, "features")


def _load_cloud(side: DictConfig, target_variable: str) -> np.ndarray:
    """Load the per-example PCA-subspace point cloud [n, k] for one side."""
    tensors = load_tensor_results(_features_dir(side, target_variable), "training_features.safetensors")
    key = "features" if "features" in tensors else next(iter(tensors))
    return tensors[key].float().cpu().numpy()


def _load_labels(side: DictConfig, target_variable: str, n: int) -> tuple[list, list]:
    """Per-example labels from the co-located train_dataset.json (same order as the features).

    Robust to correct-only filtering: the saved dataset is exactly the one that produced the
    features, so no regeneration and no ordering assumption. Returns (labels[len n], class_order).
    """
    ds_path = os.path.join(_subspace_dir(side, target_variable), "train_dataset.json")
    with open(ds_path) as fh:
        data = json.load(fh)
    labels = [ex["input"][target_variable] for ex in data]
    if len(labels) != n:
        raise ValueError(
            f"label/feature length mismatch at {ds_path}: {len(labels)} labels vs {n} feature rows."
        )
    class_order = sorted(set(labels), key=lambda v: labels.index(v))
    return labels, class_order


def _centroids(cloud: np.ndarray, labels: list, class_order: list) -> np.ndarray:
    """Per-class centroids [C, k] in `class_order` (mean of the cloud rows of each class)."""
    labels = np.asarray(labels, dtype=object)
    return np.stack([cloud[labels == c].mean(axis=0) for c in class_order], axis=0)


def _pairwise(X: np.ndarray) -> np.ndarray:
    return np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)


def _iso_scalar(iso: dict) -> float:
    """Best-effort scalar from the isometry scorer's return dict."""
    for k in ("pearson_r", "isometry", "correlation", "r", "pearson"):
        if k in iso and isinstance(iso[k], (int, float)):
            return float(iso[k])
    for v in iso.values():
        if isinstance(v, (int, float)):
            return float(v)
    return float("nan")


# --------------------------------------------------------------------------- main
def main(cfg: DictConfig) -> dict[str, Any]:
    """Align source↔target concept manifolds and score universality. Returns the metrics dict."""
    a = cfg[ANALYSIS_NAME]
    out_dir = a._output_dir
    os.makedirs(out_dir, exist_ok=True)

    tv = a.target_variable
    cloud_s = _load_cloud(a.source, tv)
    cloud_t = _load_cloud(a.target, tv)
    labels_s, classes_s = _load_labels(a.source, tv, cloud_s.shape[0])
    labels_t, classes_t = _load_labels(a.target, tv, cloud_t.shape[0])
    cent_s = _centroids(cloud_s, labels_s, classes_s)
    cent_t = _centroids(cloud_t, labels_t, classes_t)

    eps, n_init = float(a.gw.epsilon), int(a.gw.n_init)
    seed = int(a.gw.seed)
    metrics: dict[str, Any] = {
        "provenance": {
            "source": OmegaConf.to_container(a.source, resolve=True),
            "target": OmegaConf.to_container(a.target, resolve=True),
            "target_variable": a.target_variable,
            "n_classes": [len(classes_s), len(classes_t)],
            "cloud_sizes": [int(cloud_s.shape[0]), int(cloud_t.shape[0])],
            "k_features": int(cloud_s.shape[1]),
        }
    }
    coupling_to_save: dict[str, torch.Tensor] = {}

    if "centroid" in a.granularity:
        D_s, D_t = _pairwise(cent_s), _pairwise(cent_t)
        gw = gromov_wasserstein(D_s, D_t, epsilon=eps, n_init=n_init, seed=seed)
        iso = compute_isometry_metrics(D_s, D_t)
        matching = coupling_to_matching(gw["coupling"])
        corr = matching if a.procrustes.correspondence == "gw" else np.arange(len(classes_s))
        proc = procrustes_align(cent_s, cent_t, corr)
        rec = label_recovery(matching, classes_s, classes_t, symmetry_group=a.symmetry_group)
        nulls = {}
        for kind in a.nulls.kinds:
            nd = null_distribution(
                D_s, D_t, kind=kind, n_samples=int(a.nulls.n_samples), seed=int(a.nulls.seed),
                gw_epsilon=eps, gw_n_init=max(1, n_init // 2),
            )
            nd["pvalue"] = float(np.mean(np.asarray(nd["samples"]) <= gw["gw_distance"]))
            nulls[kind] = {k: v for k, v in nd.items() if k != "samples"}
        metrics["centroid"] = {
            "gw_distance": gw["gw_distance"],
            "distance_correlation": _iso_scalar(iso),
            "distance_correlation_full": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in iso.items()},
            "procrustes_residual": proc["residual"],
            "label_recovery": rec,
            "nulls": nulls,
        }
        coupling_to_save["centroid_coupling"] = torch.as_tensor(np.asarray(gw["coupling"]), dtype=torch.float32)

    if "cloud" in a.granularity:
        Dc_s, Dc_t = _pairwise(cloud_s), _pairwise(cloud_t)
        gw_c = gromov_wasserstein(Dc_s, Dc_t, epsilon=eps, n_init=max(1, n_init // 2), seed=seed)
        iso_c = compute_isometry_metrics(Dc_s, Dc_t)
        # Topology on the per-class CENTROIDS (ordered ring), not the cloud: tight per-class
        # clusters defeat Vietoris-Rips H1, but the centroid ring has a clean loop (weekdays b1=1,
        # alphabet line b1=0). See issues.md #5.
        topo_s = persistent_homology(cent_s, maxdim=int(a.topology.maxdim), rel_threshold=float(a.topology.rel_threshold))
        topo_t = persistent_homology(cent_t, maxdim=int(a.topology.maxdim), rel_threshold=float(a.topology.rel_threshold))
        gb_s = graph_betti(cent_s, k=int(a.topology.knn_k))
        gb_t = graph_betti(cent_t, k=int(a.topology.knn_k))
        metrics["cloud"] = {
            "gw_distance": gw_c["gw_distance"],
            "distance_correlation": _iso_scalar(iso_c),
            "centroid_vr_betti": {"source": topo_s["betti"], "target": topo_t["betti"],
                                  "match": betti_match(topo_s["betti"], topo_t["betti"])},
            "centroid_graph_betti": {"source": gb_s, "target": gb_t,
                                     "match": bool(gb_s["b1"] == gb_t["b1"] and gb_s["b0"] == gb_t["b0"])},
        }
        coupling_to_save["cloud_coupling"] = torch.as_tensor(np.asarray(gw_c["coupling"]), dtype=torch.float32)

    save_json_results(metrics, out_dir, "metrics.json")
    if coupling_to_save:
        save_tensor_results(coupling_to_save, out_dir, "coupling.safetensors")
    _visualize(cfg, out_dir, metrics, coupling_to_save)
    save_experiment_metadata(OmegaConf.to_container(a, resolve=True), out_dir)
    logger.info("universal_manifold[%s] → %s", a.pair_label, out_dir)
    return metrics


def _visualize(cfg: DictConfig, out_dir: str, metrics: dict, couplings: dict) -> None:
    """Coupling heatmap + isometry summary. Guarded so viz failure never kills metrics."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fmt = cfg[ANALYSIS_NAME].visualization.figure_format
        vis_dir = os.path.join(out_dir, "visualization")
        os.makedirs(vis_dir, exist_ok=True)
        if "centroid_coupling" in couplings:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(couplings["centroid_coupling"].numpy(), cmap="viridis")
            ax.set_title("GW coupling (centroids)")
            ax.set_xlabel("target class")
            ax.set_ylabel("source class")
            fig.tight_layout()
            fig.savefig(os.path.join(vis_dir, f"coupling.{fmt}"), dpi=150)
            plt.close(fig)
    except Exception as exc:  # noqa: BLE001 — viz is best-effort
        logger.warning("universal_manifold visualization skipped: %s", exc)
