"""Reference implementation of P4Transformer (Fan et al., CVPR 2021).

A clean rewrite of the original ``models/AS_p4_base.py`` from this repo,
adapted to the pn4d module layout. The architecture is byte-for-byte
equivalent so legacy checkpoints from ``train_p4.py`` can still be loaded
via :func:`pn4d.utils.load_checkpoint` with appropriate name remapping.

Pipeline (action segmentation variant):

    (B, T, N, 3)
        → P4DConv (spatio-temporal tube embedding)
        → reshape to (B, T*n, C) tokens with 4-D positional embedding
        → max-pool over points within each frame → (B, T, C)
        → Transformer encoder
        → per-frame linear head → (B, T, num_classes)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pn4d.modules.point_4d_conv import P4DConv
from pn4d.modules.transformer import TransformerEncoder


class P4Transformer(nn.Module):
    """P4Transformer for per-frame action segmentation.

    Args:
        num_classes: number of action classes.
        radius: ball-query radius used by the tube embedding.
        nsamples: number of neighbours per ball query.
        spatial_stride: FPS subsampling factor.
        temporal_kernel_size: tube length along the time axis (must be odd).
        temporal_stride: stride of the tube along time.
        emb_relu: insert a ReLU after the tube embedding.
        dim: feature dim of the transformer.
        depth: number of transformer blocks.
        heads: attention heads.
        head_dim: per-head dim (default ``dim // heads``).
        mlp_dim: hidden dim of the FFN inside each transformer block.
        causal: use a causal attention mask (online setting).
    """

    def __init__(
        self,
        num_classes: int,
        radius: float = 0.9,
        nsamples: int = 32,
        spatial_stride: int = 32,
        temporal_kernel_size: int = 3,
        temporal_stride: int = 1,
        emb_relu: bool = False,
        dim: int = 2048,
        depth: int = 5,
        heads: int = 8,
        head_dim: int = 128,
        mlp_dim: int = 1024,
        causal: bool = True,
    ) -> None:
        super().__init__()

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

        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1)
        self.emb_relu = nn.ReLU() if emb_relu else nn.Identity()

        self.transformer = TransformerEncoder(
            dim=dim,
            depth=depth,
            num_heads=heads,
            head_dim=head_dim,
            mlp_ratio=mlp_dim / dim,
            causal=causal,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, T, N, 3)`` clip.

        Returns:
            logits: ``(B, T, num_classes)``.
        """
        xyzs, features = self.tube_embedding(x)         # (B, T, N', 3), (B, T, C, N')
        features = features.transpose(2, 3)             # (B, T, N', C)
        B, T, N, C = features.shape

        # 4-D positional encoding (xyz + frame index)
        xyzts = []
        device = x.device
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
        features = features.reshape(B, T * N, C)

        xyzts = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)
        embedding = xyzts + features                                        # (B, T*N, C)

        # frame-level pooling so the transformer operates on T tokens
        embedding = embedding.reshape(B * T, N, C).permute(0, 2, 1)         # (B*T, C, N)
        frame_feat = F.adaptive_max_pool1d(embedding, 1).reshape(B, T, C)   # (B, T, C)

        frame_feat = self.emb_relu(frame_feat)
        frame_feat = self.transformer(frame_feat)                           # (B, T, C)
        return self.head(frame_feat)                                        # (B, T, num_classes)
