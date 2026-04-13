"""PointNet4D++ — P4DConv spatial backbone + reference temporal fusion.

Replaces PointNet4D's per-frame PointNet++ with P4Transformer's P4DConv
tube embedding as a more powerful spatial feature extractor. The temporal
fusion is identical to PointNet4D: depth × (Mamba3DBlock → causal Attention
→ FFN), matching the reference implementation exactly.

Architecture:

    (B, T, N, 3)
        → P4DConv tube embedding + 4-D positional encoding
        → per-frame max-pool → (B, T, C)
        → add learnable temporal positional embedding
        → depth × { Mamba3DBlock → causal Attention → FFN }
        → per-frame head → (B, T, num_classes)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pn4d.modules.point_4d_conv import P4DConv
from pn4d.modules.temporal_layers import PointNet4DTemporalFusion


class PointNet4DPlus(nn.Module):
    """P4DConv spatial encoder + reference Mamba-Transformer temporal fusion.

    Args:
        num_classes: number of action segmentation classes.
        radius: ball-query radius for the tube embedding.
        nsamples: number of spatial neighbours per ball query.
        spatial_stride: FPS subsampling factor.
        temporal_kernel_size: tube length along time (must be odd).
        temporal_stride: stride along the time axis.
        dim: feature dim (P4DConv output and temporal layer width).
        depth: number of temporal layers (Mamba + Attn + FFN each).
        heads: attention heads (default 4, per reference).
        dim_head: per-head dimension (default 1024, per reference).
        mlp_dim: FFN hidden dim (default ``dim * 4``).
        length: max sequence length for causal mask (default 150).
        head_hidden_dim: classification head hidden dim.
        max_frames: max positional embedding length.
    """

    def __init__(
        self,
        num_classes: int,
        radius: float = 0.9,
        nsamples: int = 32,
        spatial_stride: int = 32,
        temporal_kernel_size: int = 3,
        temporal_stride: int = 1,
        dim: int = 1024,
        depth: int = 5,
        heads: int = 4,
        dim_head: int = 1024,
        mlp_dim: int = 0,
        length: int = 150,
        head_hidden_dim: int = 512,
        max_frames: int = 256,
    ) -> None:
        super().__init__()
        self.dim = dim

        # 1. spatial: P4DConv tube embedding (same as P4Transformer)
        self.tube_embedding = P4DConv(
            in_planes=0,
            mlp_planes=[dim],
            mlp_batch_norm=[False],
            mlp_activation=[False],
            spatial_kernel_size=(radius, nsamples),
            spatial_stride=spatial_stride,
            temporal_kernel_size=temporal_kernel_size,
            temporal_stride=temporal_stride,
            temporal_padding=(1, 1),
            operator="+",
            spatial_pooling="max",
            temporal_pooling="max",
        )

        # 4D positional embedding: (xyz + frame_index) → dim
        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1)

        # learnable temporal positional embedding (added after frame pooling)
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, max_frames, dim))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        # 2. temporal: reference architecture — depth × (Mamba + Attn + FFN)
        self.temporal = PointNet4DTemporalFusion(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim if mlp_dim > 0 else dim * 4,
            length=length,
        )

        # 3. per-frame classification head
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return temporally-fused per-frame features ``(B, T, dim)``."""
        xyzs, features = self.tube_embedding(x)   # (B, T, N', 3), (B, T, C, N')
        features = features.transpose(2, 3)        # (B, T, N', C)
        B, T, N, C = features.shape

        # 4-D positional encoding (xyz + frame index)
        device = x.device
        xyzts = []
        xyzs_split = [s.squeeze(1).contiguous() for s in torch.split(xyzs, 1, dim=1)]
        for t, xyz in enumerate(xyzs_split):
            t_chan = torch.full(
                (xyz.size(0), xyz.size(1), 1),
                fill_value=float(t + 1),
                device=device,
                dtype=torch.float32,
            )
            xyzts.append(torch.cat([xyz, t_chan], dim=2))
        xyzts = torch.stack(xyzts, dim=1).reshape(B, T * N, 4)
        flat_features = features.reshape(B, T * N, C)

        xyzts = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)
        embedding = xyzts + flat_features  # (B, T*N, C)

        # max-pool over spatial points within each frame → (B, T, C)
        frame_feat = embedding.reshape(B * T, N, C).permute(0, 2, 1)
        frame_feat = F.adaptive_max_pool1d(frame_feat, 1).reshape(B, T, C)

        # add temporal positional embedding
        frame_feat = frame_feat + self.temporal_pos_embed[:, :T]

        return self.temporal(frame_feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        return self.head(feat)
