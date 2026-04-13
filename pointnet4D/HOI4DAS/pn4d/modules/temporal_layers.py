"""Temporal fusion layers for PointNet4D — faithful to the original implementation.

This file contains the exact Attention / FeedForward / Mamba3DBlock / causal
mask primitives used in the reference PointNet4D code, adapted to use
``mamba_ssm.Mamba`` (which is functionally equivalent to the custom bimamba
``v2`` = unidirectional mode).

The key structural difference from a standard Transformer block is that the
attention output projection includes a GELU activation, and each temporal
"layer" consists of three sequential residual sub-blocks:

    1. Mamba3DBlock:  ``x + Mamba(LN(x))``
    2. CausalAttn:    ``x + GELU(Proj(Attn(LN(x), mask)))``
    3. FFN:           ``x + FFN(LN(x))``
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from mamba_ssm import Mamba
    _HAS_MAMBA = True
except ImportError:
    Mamba = None  # type: ignore[assignment]
    _HAS_MAMBA = False


# ---------------------------------------------------------------------------
# Primitives (matching the reference transformer.py)
# ---------------------------------------------------------------------------
class Residual(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    """Multi-head attention with optional causal mask.

    NOTE: the output projection includes a **GELU** activation, matching
    the reference PointNet4D implementation.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        if mask is not None:
            mask_value = -torch.finfo(dots.dtype).max
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


def get_decoder_mask(seq_len: int) -> torch.Tensor:
    """Lower-triangular causal mask ``(seq_len, seq_len)`` of booleans."""
    return (1 - torch.triu(torch.ones((seq_len, seq_len)), diagonal=1)).bool()


# ---------------------------------------------------------------------------
# Mamba3DBlock (unidirectional, matching bimamba_type="v2")
# ---------------------------------------------------------------------------
class Mamba3DBlock(nn.Module):
    """``x + Mamba(LN(x))`` — no FFN, matching the reference implementation."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if not _HAS_MAMBA:
            raise ImportError("Mamba3DBlock requires mamba-ssm. pip install 'pn4d[mamba]'")
        self.norm = nn.LayerNorm(dim)
        self.mixer = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


# ---------------------------------------------------------------------------
# PointNet4DTemporalFusion — the complete temporal stack
# ---------------------------------------------------------------------------
class PointNet4DTemporalFusion(nn.Module):
    """Temporal fusion module for PointNet4D.

    Each of the ``depth`` layers consists of:

        1. :class:`Mamba3DBlock` — causal SSM scan
        2. :class:`Residual(PreNorm(Attention))` — causal multi-head attention
        3. :class:`Residual(PreNorm(FeedForward))` — FFN

    This exactly mirrors the reference ``PointNet4D.forward`` loop.

    Args:
        dim: feature dimension (default 1024).
        depth: number of temporal layers (default 5, matching P4T).
        heads: attention heads (default 4, per the reference).
        dim_head: per-head dimension (default 1024, per the reference).
        mlp_dim: FFN hidden dimension. If <= 0, computed as ``dim * 4``.
        length: max sequence length for the causal mask (default 150).
        dropout: dropout rate (default 0).
    """

    def __init__(
        self,
        dim: int = 1024,
        depth: int = 5,
        heads: int = 4,
        dim_head: int = 1024,
        mlp_dim: int = 0,
        length: int = 150,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if mlp_dim <= 0:
            mlp_dim = dim * 4

        # Transformer layers: depth × (Attention + FFN)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                        Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))),
                    ]
                )
            )

        # Mamba layers: depth × Mamba3DBlock
        self.mamba = nn.ModuleList([Mamba3DBlock(dim=dim) for _ in range(depth)])

        # Causal mask (pre-computed, moved to device in forward)
        self.register_buffer("mask", get_decoder_mask(length), persistent=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: ``(B, T, C)`` per-frame features.

        Returns:
            ``(B, T, C)`` temporally fused features.
        """
        T = features.size(1)
        # Ensure mask covers the sequence length (use pre-computed or generate)
        if T <= self.mask.size(0):
            mask = self.mask[:T, :T]
        else:
            mask = get_decoder_mask(T).to(features.device)

        for mamba_block, (attn, ff) in zip(self.mamba, self.layers):
            features = mamba_block(features)
            features = attn(features, mask=mask)
            features = ff(features)

        return features
