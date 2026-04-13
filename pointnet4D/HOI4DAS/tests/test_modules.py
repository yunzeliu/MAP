"""Smoke tests for the building blocks in :mod:`pn4d.modules`.

These tests use a CUDA device because pn4d's CUDA ops are CUDA-only;
they will be skipped if no GPU is available.
"""
from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
def test_pointnet_pp_extractor_forward():
    from pn4d.modules import PointNetPPExtractor

    m = PointNetPPExtractor(out_dim=128).cuda().eval()
    x = torch.randn(4, 1024, 3, device="cuda")
    with torch.no_grad():
        y = m(x)
    assert y.shape == (4, 128)
    assert torch.isfinite(y).all()


@cuda
def test_transformer_block_forward_and_grad():
    from pn4d.modules import TransformerBlock

    blk = TransformerBlock(dim=64, num_heads=4, head_dim=16).cuda().train()
    x = torch.randn(2, 16, 64, device="cuda", requires_grad=True)
    y = blk(x)
    y.sum().backward()
    assert y.shape == x.shape
    assert x.grad is not None


@cuda
def test_mamba3d_block_is_causal():
    """A causal Mamba3DBlock should yield identical outputs for prefix slices."""
    pytest.importorskip("mamba_ssm")
    from pn4d.modules import Mamba3DBlock

    blk = Mamba3DBlock(dim=64).cuda().eval()
    x = torch.randn(1, 32, 64, device="cuda")
    with torch.no_grad():
        full = blk(x)
        partial = blk(x[:, :16])
    assert torch.allclose(full[:, :16], partial, atol=1e-4)


@cuda
def test_temporal_fusion_forward():
    pytest.importorskip("mamba_ssm")
    from pn4d.modules import PointNet4DTemporalFusion

    fusion = PointNet4DTemporalFusion(dim=64, depth=2, heads=2, dim_head=32).cuda().eval()
    x = torch.randn(2, 16, 64, device="cuda")
    with torch.no_grad():
        y = fusion(x)
    assert y.shape == x.shape


@cuda
def test_p4dconv_shapes():
    from pn4d.modules import P4DConv

    conv = P4DConv(
        in_planes=0,
        mlp_planes=[32],
        mlp_batch_norm=[False],
        mlp_activation=[False],
        spatial_kernel_size=(0.5, 8),
        spatial_stride=4,
        temporal_kernel_size=3,
        temporal_padding=(1, 1),
    ).cuda().eval()
    x = torch.randn(2, 5, 64, 3, device="cuda")
    with torch.no_grad():
        new_xyz, feat = conv(x)
    assert new_xyz.shape[:3] == (2, 5, 64 // 4)
    assert feat.shape[:3] == (2, 5, 32)
