"""End-to-end forward shape tests for the full models in :mod:`pn4d.models`."""
from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
def test_p4_transformer_forward():
    from pn4d.models import P4Transformer

    m = P4Transformer(
        num_classes=19, dim=64, depth=1, heads=2, head_dim=32, mlp_dim=64
    ).cuda().eval()
    x = torch.randn(1, 8, 256, 3, device="cuda")
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, 8, 19)


@cuda
def test_pointnet4d_forward_and_grad():
    pytest.importorskip("mamba_ssm")
    from pn4d.models import PointNet4D

    m = PointNet4D(
        num_classes=19,
        feat_dim=64,
        depth=2,
        heads=2,
        dim_head=32,
    ).cuda().train()
    x = torch.randn(1, 8, 256, 3, device="cuda")
    y = m(x)
    y.sum().backward()
    assert y.shape == (1, 8, 19)


@cuda
def test_pointnet4d_extract_features():
    pytest.importorskip("mamba_ssm")
    from pn4d.models import PointNet4D

    m = PointNet4D(num_classes=19, feat_dim=64, depth=2, heads=2, dim_head=32).cuda().eval()
    x = torch.randn(1, 8, 256, 3, device="cuda")
    with torch.no_grad():
        feat = m.extract_features(x)
        logits = m(x)
    assert feat.shape == (1, 8, 64)
    assert logits.shape == (1, 8, 19)


@cuda
def test_pointnet4d_plus_forward():
    pytest.importorskip("mamba_ssm")
    from pn4d.models import PointNet4DPlus

    m = PointNet4DPlus(
        num_classes=19, dim=64, depth=2, heads=2, dim_head=32
    ).cuda().eval()
    x = torch.randn(1, 8, 256, 3, device="cuda")
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, 8, 19)


@cuda
def test_4dmap_forward_backward():
    pytest.importorskip("mamba_ssm")
    from pn4d.models import PointNet4D
    from pn4d.pretrain import FourDMAP

    enc = PointNet4D(num_classes=19, feat_dim=64, depth=1, heads=2, dim_head=32).cuda().train()
    fmap = FourDMAP(
        encoder=enc,
        feat_dim=64,
        num_points=256,
        decoder_depth=1,
        decoder_heads=2,
        mask_ratio=0.5,
    ).cuda().train()

    x = torch.randn(1, 8, 256, 3, device="cuda")
    loss, preds = fmap(x)
    loss.backward()
    assert preds.shape == (1, 8, 256, 3)
    assert torch.isfinite(loss)
