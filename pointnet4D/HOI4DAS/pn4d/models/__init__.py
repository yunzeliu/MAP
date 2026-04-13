"""Full models for 4D point-cloud video understanding.

All models follow the same call convention:

    Input:  ``(B, T, N, 3)`` — batch of clips with T frames, N points each.
    Output: ``(B, T, num_classes)`` — per-frame logits.
"""
from pn4d.models.p4_transformer import P4Transformer
from pn4d.models.pointnet4d import PointNet4D
from pn4d.models.pointnet4d_plus import PointNet4DPlus

__all__ = [
    "P4Transformer",
    "PointNet4D",
    "PointNet4DPlus",
]
