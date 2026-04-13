"""PointNet4D — lightweight 4D point cloud video backbone.

Reference: Liu et al., *PointNet4D: A Lightweight 4D Point Cloud Video
Backbone for Online and Offline Perception in Robotic Applications*,
WACV 2026.

Architecture:

    (B, T, N, 3)
        → per-frame PointNet++ → (B, T, feat_dim)
        → add learnable temporal positional embedding
        → depth × { Mamba3DBlock → causal Attention → FFN }
        → per-frame head → (B, T, num_classes)

The temporal fusion exactly follows the reference implementation:
each layer = Mamba3DBlock (no FFN) + Residual(PreNorm(Attention)) +
Residual(PreNorm(FeedForward)), with a lower-triangular causal mask
on the attention.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

from pn4d.modules.pointnet_pp_extractor import PointNetPPExtractor
from pn4d.modules.temporal_layers import PointNet4DTemporalFusion


class PointNet4D(nn.Module):
    """Per-frame PointNet++ encoder + hybrid Mamba-Transformer temporal fusion.

    Args:
        num_classes: number of action segmentation classes.
        feat_dim: feature dimension (default 1024).
        depth: number of temporal layers — each is one Mamba3DBlock +
            one causal Attention + one FFN (default 5, matching P4T depth).
        heads: attention heads (default 4, per the reference code).
        dim_head: per-head dimension (default 1024, per the reference code).
        mlp_dim: FFN hidden dimension. If ``0``, defaults to ``feat_dim * 4``.
        length: max sequence length for the causal mask (default 150).
        in_channel: extra feature channels beyond xyz (default 0).
        head_hidden_dim: classification head hidden dim (default 512).
        max_frames: max positional embedding length (default 256).
        frame_chunk: chunk size for per-frame PN++ to save memory.
        use_grad_checkpoint: wrap PN++ chunks in gradient checkpoint.
    """

    def __init__(
        self,
        num_classes: int,
        feat_dim: int = 1024,
        depth: int = 5,
        heads: int = 4,
        dim_head: int = 1024,
        mlp_dim: int = 0,
        length: int = 150,
        in_channel: int = 0,
        head_hidden_dim: int = 512,
        max_frames: int = 256,
        frame_chunk: int = 0,
        use_grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.frame_chunk = frame_chunk
        self.use_grad_checkpoint = use_grad_checkpoint

        # 1. spatial: per-frame PointNet++ → (B*T, feat_dim)
        self.frame_extractor = PointNetPPExtractor(out_dim=feat_dim, in_channel=in_channel)

        # 2. learnable temporal positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, feat_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 3. temporal: depth × (Mamba + causal Attn + FFN)
        self.temporal = PointNet4DTemporalFusion(
            dim=feat_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim if mlp_dim > 0 else feat_dim * 4,
            length=length,
        )

        # 4. per-frame classification head
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, num_classes),
        )

    @staticmethod
    def _ckpt_fn(extractor: nn.Module, chunk: torch.Tensor) -> torch.Tensor:
        return extractor(chunk)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return temporally-fused per-frame features ``(B, T, feat_dim)``."""
        B, T, N, _ = x.shape
        flat = x.reshape(B * T, N, 3)

        chunk_sz = self.frame_chunk if self.frame_chunk > 0 else flat.size(0)
        outputs: list[torch.Tensor] = []
        for start in range(0, flat.size(0), chunk_sz):
            chunk = flat[start : start + chunk_sz]
            if self.use_grad_checkpoint and self.training:
                out = ckpt.checkpoint(self._ckpt_fn, self.frame_extractor, chunk, use_reentrant=False)
            else:
                out = self.frame_extractor(chunk)
            outputs.append(out)

        per_frame = torch.cat(outputs, dim=0).reshape(B, T, self.feat_dim)
        per_frame = per_frame + self.pos_embed[:, :T]
        return self.temporal(per_frame)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, T, N, 3)`` clip.
        Returns:
            logits: ``(B, T, num_classes)``.
        """
        return self.head(self.extract_features(x))
