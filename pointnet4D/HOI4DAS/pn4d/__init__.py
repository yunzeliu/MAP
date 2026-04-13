"""
pn4d — PointNet4D and friends for 4D point cloud video understanding.

Public sub-modules:
    - pn4d.ops      : low-level point cloud ops (FPS, ball query, …)
    - pn4d.modules  : reusable nn.Module building blocks
    - pn4d.models   : full models (P4Transformer, PointNet4D, …)
    - pn4d.pretrain : self-supervised pretraining (4DMAP)
    - pn4d.data     : datasets (HOI4D action segmentation, …)
    - pn4d.engine   : training / evaluation loops
    - pn4d.utils    : metrics, schedulers, logging helpers

Most users only need:

    >>> import pn4d
    >>> model = pn4d.models.PointNet4D(num_classes=19)
    >>> dataset = pn4d.data.HOI4DActionSeg(root="...", split="train")
"""
from __future__ import annotations

__version__ = "0.1.0"

# Re-export the most common entry points so that ``import pn4d`` is enough.
from pn4d import data, engine, models, modules, pretrain, utils

__all__ = [
    "__version__",
    "data",
    "engine",
    "models",
    "modules",
    "pretrain",
    "utils",
]
