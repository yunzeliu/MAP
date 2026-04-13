"""Single-frame PointNet++ feature extractor used by :class:`pn4d.models.PointNet4D`.

This is a lightweight 2-stage Set-Abstraction encoder that maps a single
frame ``(N, 3)`` to a global feature vector of dimension ``out_dim``. It
follows the standard SSG variant of PointNet++ (Qi et al., NeurIPS 2017)
and uses pn4d's bundled CUDA ops for FPS / ball query / grouping.

The extractor intentionally avoids extra abstractions (no SharedMLP /
``pytorch_utils.Conv2d`` wrapper) so the code stays self-contained and
easy to read for users coming from other PointNet++ codebases.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pn4d.ops import (
    QueryAndGroup,
    furthest_point_sample,
    gather_operation,
)


class _SAModuleSSG(nn.Module):
    """Single-scale Set Abstraction module."""

    def __init__(
        self,
        in_channel: int,
        npoint: Optional[int],
        radius: float,
        nsample: int,
        mlp: List[int],
        use_xyz: bool = True,
    ) -> None:
        super().__init__()
        self.npoint = npoint
        self.grouper = QueryAndGroup(radius=radius, nsample=nsample, use_xyz=use_xyz)

        # in_channel here already accounts for the +3 added by ``use_xyz`` if applicable
        c_in = in_channel + (3 if use_xyz else 0)
        layers: List[nn.Module] = []
        for c_out in mlp:
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.ReLU(inplace=True))
            c_in = c_out
        self.mlp = nn.Sequential(*layers)
        self.out_channel = mlp[-1]

    def forward(
        self,
        xyz: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: ``(B, N, 3)`` xyz coordinates.
            features: ``(B, C, N)`` features or ``None``.

        Returns:
            new_xyz: ``(B, npoint, 3)`` (or input ``xyz`` if npoint is None).
            new_features: ``(B, mlp[-1], npoint)``.
        """
        if self.npoint is not None:
            xyz_t = xyz.transpose(1, 2).contiguous()
            fps_idx = furthest_point_sample(xyz, self.npoint)
            new_xyz = gather_operation(xyz_t, fps_idx).transpose(1, 2).contiguous()
        else:
            new_xyz = xyz

        grouped = self.grouper(xyz, new_xyz, features)  # (B, C', npoint, nsample)
        feats = self.mlp(grouped)
        feats = feats.max(dim=-1).values  # (B, mlp[-1], npoint)
        return new_xyz, feats


class PointNetPPExtractor(nn.Module):
    """Per-frame PointNet++ encoder.

    Default config (matches the SSG variant of Qi et al. 2017 with the
    output channel set to ``out_dim``):

        SA1: npoint=512, radius=0.2, nsample=32, mlp=[64, 64, 128]
        SA2: npoint=128, radius=0.4, nsample=64, mlp=[128, 128, 256]
        SA3: global,                              mlp=[256, 512, out_dim]

    Args:
        out_dim: dimensionality of the per-frame feature vector.
        in_channel: number of feature channels in addition to xyz
            (set to ``0`` for xyz-only inputs).
    """

    def __init__(self, out_dim: int = 256, in_channel: int = 0) -> None:
        super().__init__()
        self.sa1 = _SAModuleSSG(
            in_channel=in_channel,
            npoint=512,
            radius=0.2,
            nsample=32,
            mlp=[64, 64, 128],
        )
        self.sa2 = _SAModuleSSG(
            in_channel=128,
            npoint=128,
            radius=0.4,
            nsample=64,
            mlp=[128, 128, 256],
        )
        # global SA: gather all points around a single dummy centre
        self.sa3 = _SAModuleSSG(
            in_channel=256,
            npoint=1,
            radius=1e9,
            nsample=128,
            mlp=[256, 512, out_dim],
            use_xyz=True,
        )
        self.out_dim = out_dim

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: ``(B, N, 3)`` single-frame point cloud.

        Returns:
            feature: ``(B, out_dim)`` global frame feature.
        """
        l1_xyz, l1_feat = self.sa1(xyz, None)
        l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat)
        _, l3_feat = self.sa3(l2_xyz, l2_feat)  # (B, out_dim, 1)
        return l3_feat.squeeze(-1)
