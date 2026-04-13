"""4DMAP — frame-level Masked Autoregressive Pretraining for PointNet4D.

Reference: Liu et al., *PointNet4D* (WACV 2026), Section 4.

Algorithm:

    1. Sample a fraction (default 50 %) of frames at random.
    2. Run the backbone on the entire clip to obtain per-frame features
       ``F = (B, T, C)``.
    3. Replace the *masked* frame features with a learnable mask token
       and feed the resulting sequence to a causal Transformer decoder
       that autoregressively predicts the original masked frames as a
       set of N xyz points.
    4. Loss: Chamfer-Distance L2 between predicted and ground-truth
       point clouds, averaged over masked frames.

The encoder of choice is :class:`pn4d.models.PointNet4D` exposed via
its ``extract_features`` method, but any module with the same interface
(``forward(x: (B, T, N, 3)) -> (B, T, C)``) can be plugged in.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from pn4d.pretrain.decoder import ARTransformerDecoder


def chamfer_distance_l2(
    pred: torch.Tensor,
    gt: torch.Tensor,
    chunk_size: int = 2,
) -> torch.Tensor:
    """Symmetric Chamfer-Distance L2 over batched point clouds.

    Computes the loss in mini-batches along the leading batch dim so the
    pairwise distance matrix never exceeds ``chunk_size * P * M`` floats.
    Uses :func:`torch.cdist` (cuBLAS GEMM) which is far cheaper than the
    explicit ``unsqueeze`` / ``diff.pow(2).sum`` formulation.

    Args:
        pred: ``(*, P, 3)`` predicted points.
        gt:   ``(*, M, 3)`` ground-truth points (same leading dims).
        chunk_size: number of leading items per cdist call.

    Returns:
        scalar mean CD-L2.
    """
    if pred.dim() < 2 or gt.dim() < 2:
        raise ValueError("pred / gt must have at least 2 dims (..., P, 3)")

    P = pred.shape[-2]
    M = gt.shape[-2]
    flat_pred = pred.reshape(-1, P, 3)
    flat_gt = gt.reshape(-1, M, 3)
    B = flat_pred.shape[0]
    if B == 0:
        return pred.sum() * 0  # gradient-friendly zero

    cd_terms: list[torch.Tensor] = []
    for start in range(0, B, chunk_size):
        end = start + chunk_size
        a = flat_pred[start:end]
        b = flat_gt[start:end]
        # squared L2 = cdist(a, b) ** 2; cdist itself is L2 distance.
        d2 = torch.cdist(a, b, p=2.0).pow(2)            # (b, P, M)
        p2g = d2.min(dim=2).values.mean(dim=-1)         # (b,)
        g2p = d2.min(dim=1).values.mean(dim=-1)         # (b,)
        cd_terms.append(p2g + g2p)
    return torch.cat(cd_terms, dim=0).mean()


def sample_frame_mask(
    batch_size: int,
    num_frames: int,
    mask_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """Random per-clip frame mask. Returns ``(B, T)`` boolean tensor."""
    if not 0 < mask_ratio < 1:
        raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
    num_mask = max(1, int(round(num_frames * mask_ratio)))
    mask = torch.zeros(batch_size, num_frames, dtype=torch.bool, device=device)
    for b in range(batch_size):
        idx = torch.randperm(num_frames, device=device)[:num_mask]
        mask[b, idx] = True
    return mask


class FourDMAP(nn.Module):
    """4D Masked Autoregressive Pretraining wrapper.

    Args:
        encoder: a backbone exposing ``extract_features(x) -> (B, T, C)``
            (e.g. :class:`pn4d.models.PointNet4D`). The encoder must keep
            the temporal dimension and accept ``(B, T, N, 3)`` inputs.
        feat_dim: feature dim ``C`` produced by the encoder.
        num_points: number of points per frame to reconstruct (= ``N``).
        decoder_depth: depth of the AR Transformer decoder.
        decoder_heads: heads of the AR Transformer decoder.
        mask_ratio: fraction of frames to mask (default 0.5 from the paper).
        max_frames: max sequence length the decoder caches positional emb.
    """

    def __init__(
        self,
        encoder: nn.Module,
        feat_dim: int,
        num_points: int,
        decoder_depth: int = 4,
        decoder_heads: int = 8,
        mask_ratio: float = 0.5,
        max_frames: int = 256,
    ) -> None:
        super().__init__()
        if not hasattr(encoder, "extract_features"):
            raise AttributeError(
                "encoder must implement `extract_features(x) -> (B, T, C)`"
            )
        self.encoder = encoder
        self.feat_dim = feat_dim
        self.num_points = num_points
        self.mask_ratio = mask_ratio

        self.decoder = ARTransformerDecoder(
            feat_dim=feat_dim,
            num_points=num_points,
            depth=decoder_depth,
            num_heads=decoder_heads,
            max_frames=max_frames,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: ``(B, T, N, 3)`` clip.
            mask: optional ``(B, T)`` boolean mask. If ``None``, a fresh
                random mask is sampled with ``self.mask_ratio``.

        Returns:
            loss: scalar CD-L2 reconstruction loss on masked frames.
            preds: ``(B, T, N, 3)`` decoder outputs (for inspection).
        """
        B, T, N, _ = x.shape
        if mask is None:
            mask = sample_frame_mask(B, T, self.mask_ratio, x.device)
        else:
            assert mask.shape == (B, T) and mask.dtype == torch.bool

        feat = self.encoder.extract_features(x)            # (B, T, C)
        preds = self.decoder(feat, mask)                   # (B, T, N, 3)

        gt_masked = x[mask]                                 # (Nmask_total, N, 3)
        pred_masked = preds[mask]                           # (Nmask_total, N, 3)
        loss = chamfer_distance_l2(pred_masked, gt_masked)
        return loss, preds
