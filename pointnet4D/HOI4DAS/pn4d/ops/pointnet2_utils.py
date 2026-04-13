"""
Differentiable point-cloud ops backed by the bundled CUDA kernels in
``pn4d._C``.

This is a streamlined re-implementation of the well-known
`Pointnet2_PyTorch <https://github.com/erikwijmans/Pointnet2_PyTorch>`_
layer, kept on parity with the API used by the rest of pn4d.

The original PointNet++ wrappers from the legacy ``modules/`` package
require an external ``pytorch_utils`` helper for ``Conv2d``/``BN``; we
intentionally drop that here so the ``ops`` sub-package only depends on
plain PyTorch.

All Function classes accept and return torch tensors; the typical input
is ``xyz`` of shape ``(B, N, 3)`` (channels-last) and ``features`` of
shape ``(B, C, N)`` (channels-first), matching the legacy convention.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.autograd import Function

from pn4d import _C  # built by setup.py — pn4d/_C*.so


# ---------------------------------------------------------------------------
# Furthest point sampling
# ---------------------------------------------------------------------------
class FurthestPointSampling(Function):
    """Iterative furthest-point sampling.

    Args:
        xyz: ``(B, N, 3)`` xyz coordinates.
        npoint: number of points to sample.

    Returns:
        ``(B, npoint)`` int32 indices into ``xyz``.
    """

    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        fps_inds = _C.furthest_point_sampling(xyz, npoint)
        ctx.mark_non_differentiable(fps_inds)
        return fps_inds

    @staticmethod
    def backward(ctx, *grads):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


# ---------------------------------------------------------------------------
# Gather operation (index features by FPS indices)
# ---------------------------------------------------------------------------
class GatherOperation(Function):
    """Gather features along the points dim.

    Args:
        features: ``(B, C, N)`` features.
        idx: ``(B, npoint)`` int32 indices.

    Returns:
        ``(B, C, npoint)`` gathered features.
    """

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        _, C, N = features.size()
        ctx.for_backwards = (idx, C, N)
        return _C.gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        idx, C, N = ctx.for_backwards
        grad_features = _C.gather_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None


gather_operation = GatherOperation.apply


# ---------------------------------------------------------------------------
# Three nearest neighbours
# ---------------------------------------------------------------------------
class ThreeNN(Function):
    """Find the three nearest neighbours from ``known`` for each ``unknown``.

    Returns ``(dist, idx)`` where ``dist`` is the L2 distance.
    """

    @staticmethod
    def forward(ctx, unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist2, idx = _C.three_nn(unknown, known)
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, *grads):
        return None, None


three_nn = ThreeNN.apply


# ---------------------------------------------------------------------------
# Three interpolate (used by feature propagation)
# ---------------------------------------------------------------------------
class ThreeInterpolate(Function):
    """Weighted linear interpolation across the 3 nearest neighbours."""

    @staticmethod
    def forward(
        ctx,
        features: torch.Tensor,
        idx: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        _, _, m = features.size()
        ctx.three_interpolate_for_backward = (idx, weight, m)
        return _C.three_interpolate(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        idx, weight, m = ctx.three_interpolate_for_backward
        grad_features = _C.three_interpolate_grad(grad_out.contiguous(), idx, weight, m)
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


# ---------------------------------------------------------------------------
# Grouping (gather neighbours for SA modules)
# ---------------------------------------------------------------------------
class GroupingOperation(Function):
    """Gather neighbour features given pre-computed grouping indices."""

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        _, _, N = features.size()
        ctx.for_backwards = (idx, N)
        return _C.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        idx, N = ctx.for_backwards
        grad_features = _C.group_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None


grouping_operation = GroupingOperation.apply


# ---------------------------------------------------------------------------
# Ball query
# ---------------------------------------------------------------------------
class BallQuery(Function):
    """Group points within a radius around each query."""

    @staticmethod
    def forward(
        ctx,
        radius: float,
        nsample: int,
        xyz: torch.Tensor,
        new_xyz: torch.Tensor,
    ) -> torch.Tensor:
        inds = _C.ball_query(new_xyz, xyz, radius, nsample)
        ctx.mark_non_differentiable(inds)
        return inds

    @staticmethod
    def backward(ctx, *grads):
        return None, None, None, None


ball_query = BallQuery.apply


# ---------------------------------------------------------------------------
# nn.Module wrappers
# ---------------------------------------------------------------------------
class QueryAndGroup(nn.Module):
    """Group points around ``new_xyz`` centres via ball query.

    Returns ``(B, 3+C, npoint, nsample)`` features (xyz delta concatenated
    with input features). If ``features`` is ``None``, only the centred xyz
    is returned.
    """

    def __init__(
        self,
        radius: float,
        nsample: int,
        use_xyz: bool = True,
        normalize_xyz: bool = False,
    ) -> None:
        super().__init__()
        self.radius = radius
        self.nsample = nsample
        self.use_xyz = use_xyz
        self.normalize_xyz = normalize_xyz

    def forward(
        self,
        xyz: torch.Tensor,
        new_xyz: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_t = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_t, idx)
        grouped_xyz = grouped_xyz - new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz = grouped_xyz / self.radius

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                return torch.cat([grouped_xyz, grouped_features], dim=1)
            return grouped_features
        if not self.use_xyz:
            raise ValueError("If features is None then use_xyz must be True")
        return grouped_xyz


class GroupAll(nn.Module):
    """Group all points (no neighbourhood selection)."""

    def __init__(self, use_xyz: bool = True) -> None:
        super().__init__()
        self.use_xyz = use_xyz

    def forward(
        self,
        xyz: torch.Tensor,
        new_xyz: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                return torch.cat([grouped_xyz, grouped_features], dim=1)
            return grouped_features
        return grouped_xyz
