"""Session-local method `cross_model_alignment` (see set_up_method.md)."""

from .cross_model_alignment import (
    coupling_to_matching,
    gromov_wasserstein,
    label_recovery,
    null_distribution,
    procrustes_align,
)

__all__ = [
    "gromov_wasserstein",
    "procrustes_align",
    "coupling_to_matching",
    "label_recovery",
    "null_distribution",
]
