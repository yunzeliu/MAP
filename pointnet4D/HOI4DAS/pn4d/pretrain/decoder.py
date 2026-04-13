"""Causal Transformer decoder for autoregressive 4D point cloud reconstruction.

The decoder consumes the temporal feature sequence produced by a pn4d
backbone (e.g. :class:`pn4d.models.PointNet4D`) and predicts the masked
frames *one at a time*, attending only to preceding visible frames.

For 4DMAP we predict whole frames (``N`` points each) at once rather
than going point-by-point — the paper notes that frame-wise decoding is
both more efficient and more aligned with the temporal nature of the
masking strategy.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from pn4d.modules.transformer import TransformerEncoder


class ARTransformerDecoder(nn.Module):
    """Causal Transformer decoder for frame-wise AR reconstruction.

    The decoder is intentionally implemented as a stack of *causal*
    Transformer encoder blocks (with an upper-triangular self-attention
    mask) rather than a true encoder-decoder pair. This matches the
    4DMAP paper's design and keeps the implementation simple.

    Args:
        feat_dim: encoder feature dim (= input token dim of the decoder).
        num_points: number of xyz points to predict per frame.
        depth: number of decoder transformer blocks.
        num_heads: number of attention heads per block.
        mlp_ratio: FFN hidden ratio.
        max_frames: maximum sequence length the decoder will see; only
            used to cache the learned positional embedding (default 256,
            HOI4D clips have 150 frames).
    """

    def __init__(
        self,
        feat_dim: int,
        num_points: int,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        max_frames: int = 256,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.num_points = num_points

        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, feat_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.blocks = TransformerEncoder(
            dim=feat_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            causal=True,
        )

        self.norm = nn.LayerNorm(feat_dim)
        # Predict all N points of the masked frame from a single token.
        self.to_points = nn.Linear(feat_dim, num_points * 3)

    def forward(
        self,
        feat: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feat: ``(B, T, feat_dim)`` per-frame features from the encoder.
                Frames flagged by ``mask`` should already be replaced with
                the mask token by the caller, OR this method does it.
            mask: ``(B, T)`` boolean tensor; ``True`` = masked frame.

        Returns:
            preds: ``(B, T, N, 3)`` predicted xyz for every frame. Only
                the frames where ``mask == True`` are used by the loss.
        """
        B, T, C = feat.shape
        assert C == self.feat_dim, f"feat dim mismatch: {C} vs {self.feat_dim}"
        assert mask.shape == (B, T)

        # 1. replace masked positions with the learnable mask token
        mask_tok = self.mask_token.expand(B, T, C)
        h = torch.where(mask.unsqueeze(-1), mask_tok, feat)

        # 2. add positional embedding (truncated to T)
        h = h + self.pos_embed[:, :T]

        # 3. causal transformer
        h = self.blocks(h)
        h = self.norm(h)

        # 4. project to point cloud
        points = self.to_points(h).reshape(B, T, self.num_points, 3)
        return points
