"""4D point convolution from `P4Transformer (CVPR 2021)
<https://openaccess.thecvf.com/content/CVPR2021/papers/Fan_Point_4D_Transformer_Networks_for_Spatio-Temporal_Modeling_in_Point_Cloud_CVPR_2021_paper.pdf>`_.

Cleaned up version of the legacy ``modules/point_4d_convolution.py``:
    - imports from :mod:`pn4d.ops` so the bundled CUDA extension is used,
    - drops the ``sys.path.append`` hack,
    - keeps the math byte-for-byte identical so existing checkpoints load.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from pn4d.ops import (
    ball_query,
    furthest_point_sample,
    gather_operation,
    grouping_operation,
    three_interpolate,
    three_nn,
)


class P4DConv(nn.Module):
    """Spatio-temporal point convolution that aggregates a local
    spatio-temporal tube around each anchor point.

    The block subsamples each frame to ``N // spatial_stride`` anchor
    points (FPS), then for every neighbouring frame in the temporal kernel
    queries ``k`` neighbours within radius ``r`` and aggregates them with
    a small MLP. Finally features are pooled over the temporal kernel.

    Args:
        in_planes: number of input feature channels (0 for ``xyz``-only).
        mlp_planes: hidden dims of the per-tube MLP.
        mlp_batch_norm: whether to apply BN after each MLP layer.
        mlp_activation: whether to apply ReLU after each MLP layer.
        spatial_kernel_size: ``(radius, k)`` for the ball query.
        spatial_stride: subsampling rate of FPS.
        temporal_kernel_size: must be odd, defines the temporal tube length.
        temporal_stride: stride along the time axis.
        temporal_padding: ``(left, right)`` extra frames to pad.
        temporal_padding_mode: ``"replicate"`` or ``"zeros"``.
        operator: how feature and displacement embeddings combine,
            ``"+"`` (default) or ``"*"``.
        spatial_pooling: ``"max"``, ``"sum"`` or ``"mean"`` over neighbours.
        temporal_pooling: ``"max"``, ``"sum"`` or ``"mean"`` over the tube.
        bias: whether the per-MLP convs use bias.
    """

    def __init__(
        self,
        in_planes: int,
        mlp_planes: List[int],
        mlp_batch_norm: List[bool],
        mlp_activation: List[bool],
        spatial_kernel_size: Tuple[float, int],
        spatial_stride: int,
        temporal_kernel_size: int,
        temporal_stride: int = 1,
        temporal_padding: Tuple[int, int] = (0, 0),
        temporal_padding_mode: str = "replicate",
        operator: str = "+",
        spatial_pooling: str = "max",
        temporal_pooling: str = "sum",
        bias: bool = False,
    ) -> None:
        super().__init__()

        if temporal_kernel_size % 2 != 1:
            raise ValueError("P4DConv: temporal_kernel_size must be odd")

        self.in_planes = in_planes
        self.mlp_planes = mlp_planes
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_activation = mlp_activation

        self.r, self.k = spatial_kernel_size
        self.spatial_stride = spatial_stride

        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_stride = temporal_stride
        self.temporal_padding = list(temporal_padding)
        self.temporal_padding_mode = temporal_padding_mode

        self.operator = operator
        self.spatial_pooling = spatial_pooling
        self.temporal_pooling = temporal_pooling

        # 4-channel displacement (dx, dy, dz, dt) embedding
        conv_d: List[nn.Module] = [
            nn.Conv2d(4, mlp_planes[0], kernel_size=1, bias=bias)
        ]
        if mlp_batch_norm[0]:
            conv_d.append(nn.BatchNorm2d(mlp_planes[0]))
        if mlp_activation[0]:
            conv_d.append(nn.ReLU(inplace=True))
        self.conv_d = nn.Sequential(*conv_d)

        # optional input-feature embedding
        if in_planes != 0:
            conv_f: List[nn.Module] = [
                nn.Conv2d(in_planes, mlp_planes[0], kernel_size=1, bias=bias)
            ]
            if mlp_batch_norm[0]:
                conv_f.append(nn.BatchNorm2d(mlp_planes[0]))
            if mlp_activation[0]:
                conv_f.append(nn.ReLU(inplace=True))
            self.conv_f = nn.Sequential(*conv_f)

        # post-aggregation MLP
        mlp: List[nn.Module] = []
        for i in range(1, len(mlp_planes)):
            if mlp_planes[i] != 0:
                mlp.append(nn.Conv2d(mlp_planes[i - 1], mlp_planes[i], kernel_size=1, bias=bias))
            if mlp_batch_norm[i]:
                mlp.append(nn.BatchNorm2d(mlp_planes[i]))
            if mlp_activation[i]:
                mlp.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*mlp)

    def forward(
        self,
        xyzs: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyzs: ``(B, T, N, 3)`` xyz coordinates per frame.
            features: ``(B, T, C, N)`` per-point features (or ``None``).

        Returns:
            new_xyzs: ``(B, T', N', 3)`` subsampled anchor positions.
            new_features: ``(B, T', C', N')`` aggregated tube features.
        """
        device = xyzs.device
        nframes = xyzs.size(1)
        npoints = xyzs.size(2)

        if (nframes + sum(self.temporal_padding) - self.temporal_kernel_size) % self.temporal_stride != 0:
            raise ValueError("P4DConv: temporal padding/stride mismatch with input length")

        xyz_list = [t.contiguous() for t in torch.split(xyzs, 1, dim=1)]
        xyz_list = [x.squeeze(1) for x in xyz_list]

        if self.temporal_padding_mode == "zeros":
            zero_xyz = torch.zeros_like(xyz_list[0])
            xyz_list = [zero_xyz] * self.temporal_padding[0] + xyz_list + [zero_xyz] * self.temporal_padding[1]
        else:  # replicate
            xyz_list = (
                [xyz_list[0]] * self.temporal_padding[0]
                + xyz_list
                + [xyz_list[-1]] * self.temporal_padding[1]
            )

        feat_list: Optional[List[torch.Tensor]] = None
        if self.in_planes != 0:
            assert features is not None
            feat_list = [t.contiguous() for t in torch.split(features, 1, dim=1)]
            feat_list = [f.squeeze(1) for f in feat_list]
            if self.temporal_padding_mode == "zeros":
                zero_f = torch.zeros_like(feat_list[0])
                feat_list = (
                    [zero_f] * self.temporal_padding[0]
                    + feat_list
                    + [zero_f] * self.temporal_padding[1]
                )
            else:
                feat_list = (
                    [feat_list[0]] * self.temporal_padding[0]
                    + feat_list
                    + [feat_list[-1]] * self.temporal_padding[1]
                )

        new_xyzs: List[torch.Tensor] = []
        new_features: List[torch.Tensor] = []
        half = self.temporal_kernel_size // 2

        for t in range(half, len(xyz_list) - half, self.temporal_stride):
            anchor_idx = furthest_point_sample(xyz_list[t], npoints // self.spatial_stride)
            anchor_xyz_flipped = gather_operation(
                xyz_list[t].transpose(1, 2).contiguous(), anchor_idx
            )  # (B, 3, N')
            anchor_xyz_expanded = anchor_xyz_flipped.unsqueeze(3)  # (B, 3, N', 1)
            anchor_xyz = anchor_xyz_flipped.transpose(1, 2).contiguous()  # (B, N', 3)

            tube_features: List[torch.Tensor] = []
            for i in range(t - half, t + half + 1):
                neighbor_xyz = xyz_list[i]

                idx = ball_query(self.r, self.k, neighbor_xyz, anchor_xyz)
                neighbor_xyz_flipped = neighbor_xyz.transpose(1, 2).contiguous()
                neighbor_xyz_grouped = grouping_operation(neighbor_xyz_flipped, idx)  # (B, 3, N', k)

                xyz_displacement = neighbor_xyz_grouped - anchor_xyz_expanded
                t_displacement = torch.full(
                    (xyz_displacement.size(0), 1, xyz_displacement.size(2), xyz_displacement.size(3)),
                    fill_value=float(i - t),
                    dtype=torch.float32,
                    device=device,
                )
                displacement = torch.cat((xyz_displacement, t_displacement), dim=1)
                displacement = self.conv_d(displacement)

                if self.in_planes != 0:
                    assert feat_list is not None
                    neighbor_feature_grouped = grouping_operation(feat_list[i], idx)
                    feature = self.conv_f(neighbor_feature_grouped)
                    feature = feature + displacement if self.operator == "+" else feature * displacement
                else:
                    feature = displacement

                feature = self.mlp(feature)
                if self.spatial_pooling == "max":
                    feature = feature.max(dim=-1).values
                elif self.spatial_pooling == "sum":
                    feature = feature.sum(dim=-1)
                else:
                    feature = feature.mean(dim=-1)

                tube_features.append(feature)

            stacked = torch.stack(tube_features, dim=1)
            if self.temporal_pooling == "max":
                merged = stacked.max(dim=1).values
            elif self.temporal_pooling == "sum":
                merged = stacked.sum(dim=1)
            else:
                merged = stacked.mean(dim=1)

            new_xyzs.append(anchor_xyz)
            new_features.append(merged)

        return torch.stack(new_xyzs, dim=1), torch.stack(new_features, dim=1)


class P4DTransConv(nn.Module):
    """Inverse of :class:`P4DConv` — feature propagation via 3-NN interpolation,
    used by encoder-decoder segmentation models.
    """

    def __init__(
        self,
        in_planes: int,
        mlp_planes: List[int],
        mlp_batch_norm: List[bool],
        mlp_activation: List[bool],
        original_planes: int = 0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_planes = in_planes
        self.mlp_planes = mlp_planes

        layers: List[nn.Module] = []
        for i, planes in enumerate(mlp_planes):
            in_c = (in_planes + original_planes) if i == 0 else mlp_planes[i - 1]
            layers.append(nn.Conv1d(in_c, planes, kernel_size=1, bias=bias))
            if mlp_batch_norm[i]:
                layers.append(nn.BatchNorm1d(planes))
            if mlp_activation[i]:
                layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(
        self,
        xyzs: torch.Tensor,
        original_xyzs: torch.Tensor,
        features: torch.Tensor,
        original_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T = xyzs.size(1)
        xyz_list = [x.squeeze(1).contiguous() for x in torch.split(xyzs, 1, dim=1)]
        feat_list = [f.squeeze(1).contiguous() for f in torch.split(features, 1, dim=1)]

        new_xyzs = original_xyzs
        original_xyz_list = [x.squeeze(1).contiguous() for x in torch.split(original_xyzs, 1, dim=1)]
        if original_features is not None:
            original_feat_list = [
                f.squeeze(1).contiguous() for f in torch.split(original_features, 1, dim=1)
            ]
        else:
            original_feat_list = None  # type: ignore[assignment]

        outputs: List[torch.Tensor] = []
        for t in range(T):
            dist, idx = three_nn(original_xyz_list[t], xyz_list[t])
            dist_recip = 1.0 / (dist + 1e-8)
            norm = dist_recip.sum(dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated = three_interpolate(feat_list[t], idx, weight)
            if original_feat_list is not None:
                interpolated = torch.cat([interpolated, original_feat_list[t]], dim=1)
            outputs.append(self.conv(interpolated))

        return new_xyzs, torch.stack(outputs, dim=1)
