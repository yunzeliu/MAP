"""Evaluation loop and segmentation metrics aggregator."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pn4d.utils.logger import MetricLogger
from pn4d.utils.metrics import edit_score, f_score


def segmentation_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    overlaps: Sequence[float] = (0.1, 0.25, 0.5),
    bg_class: Sequence = ("background",),
) -> Dict[str, float]:
    """Compute frame-wise accuracy, edit score and F1@k.

    Args:
        preds: ``(N_clips, T)`` integer predictions.
        targets: ``(N_clips, T)`` ground-truth integer labels.
        overlaps: IoU thresholds at which to compute F1.
        bg_class: labels treated as background and excluded from segments.

    Returns:
        dict with keys: ``acc``, ``edit``, and ``f1@{overlap}``.
    """
    assert preds.shape == targets.shape, f"shape mismatch {preds.shape} vs {targets.shape}"
    n_clips = preds.shape[0]

    acc = float((preds == targets).mean())
    edit = 0.0
    tps = np.zeros(len(overlaps))
    fps = np.zeros(len(overlaps))
    fns = np.zeros(len(overlaps))

    for b in range(n_clips):
        edit += edit_score(preds[b], targets[b], bg_class=bg_class)
        for s, ov in enumerate(overlaps):
            tp1, fp1, fn1 = f_score(preds[b], targets[b], ov, bg_class=bg_class)
            tps[s] += tp1
            fps[s] += fp1
            fns[s] += fn1

    out: Dict[str, float] = {"acc": acc, "edit": edit / max(n_clips, 1)}
    for s, ov in enumerate(overlaps):
        precision = tps[s] / max(tps[s] + fps[s], 1.0)
        recall = tps[s] / max(tps[s] + fns[s], 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        out[f"f1@{ov:.2f}"] = float(np.nan_to_num(f1) * 100)
    return out


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: Optional[nn.Module],
    data_loader: DataLoader,
    device: torch.device,
    overlaps: Sequence[float] = (0.1, 0.25, 0.5),
    print_freq: int = 20,
) -> Dict[str, float]:
    """Run a full pass over the test loader and return metrics.

    Predictions and targets are concatenated across the loader so that
    the segmentation metrics are computed once at the end (matching the
    convention of MS-TCN / P4Transformer).
    """
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    all_pred: list[np.ndarray] = []
    all_target: list[np.ndarray] = []

    for clip, target in metric_logger.log_every(data_loader, print_freq, header):
        clip = clip.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(clip)                                # (B, T, K)
        if criterion is not None:
            loss = criterion(output.permute(0, 2, 1), target)
            metric_logger.update(loss=loss.item())

        pred = output.argmax(dim=-1)
        all_pred.append(pred.cpu().numpy().astype(np.int32))
        all_target.append(target.cpu().numpy().astype(np.int32))

    metric_logger.synchronize_between_processes()
    preds = np.concatenate(all_pred, axis=0)
    targets = np.concatenate(all_target, axis=0)
    metrics = segmentation_metrics(preds, targets, overlaps=overlaps)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    return metrics
