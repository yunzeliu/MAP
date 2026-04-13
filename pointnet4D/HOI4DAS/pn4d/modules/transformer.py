"""Standard pre-LN Transformer encoder.

Both :class:`TransformerBlock` and :class:`TransformerEncoder` mimic the
ViT layout (PreNorm → Attention → residual → PreNorm → MLP → residual).
A causal flag enables online (unidirectional) attention so the same
module can be used in both online and offline settings of pn4d.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class _Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if head_dim is None:
            assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
            head_dim = dim // num_heads
        inner_dim = head_dim * num_heads

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.proj = nn.Linear(inner_dim, dim)
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop if self.training else 0.0, is_causal=causal
        )
        out = out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block: ``x + Attn(LN(x))`` then ``x + MLP(LN(x))``.

    Args:
        dim: feature dimension.
        num_heads: number of attention heads.
        head_dim: per-head dim (default ``dim // num_heads``).
        mlp_ratio: hidden ratio for the FFN.
        causal: if True, attention uses an upper-triangular mask
            (online setting).
        attn_drop, proj_drop, mlp_drop: dropout rates.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        mlp_ratio: float = 4.0,
        causal: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.causal = causal
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(
            dim,
            num_heads=num_heads,
            head_dim=head_dim,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio), dropout=mlp_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), causal=self.causal)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Stack of :class:`TransformerBlock`\\ s."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        mlp_ratio: float = 4.0,
        causal: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    causal=causal,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    mlp_drop=mlp_drop,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x
