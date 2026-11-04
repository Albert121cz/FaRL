# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .common import (load_checkpoint, Activation, MLP, Residual)
from .geometry import (normalize_points, denormalize_points,
                       points2heatmap, heatmap2points)
from .transformers import FaRLVisualFeatures

__all__ = [
    "load_checkpoint", "Activation", "MLP", "Residual",
    "normalize_points", "denormalize_points", "points2heatmap",
    "heatmap2points", "MMSEG_UPerHead", "FaRLVisualFeatures",
]


def __getattr__(name):
    if name == "MMSEG_UPerHead":
        from .mmseg import MMSEG_UPerHead
        return MMSEG_UPerHead
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
