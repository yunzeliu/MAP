"""Single-epoch training loop for action segmentation models."""
from __future__ import annotations

import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pn4d.utils.logger import MetricLogger, SmoothedValue


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    print_freq: int = 20,
) -> None:
    """Run one epoch over ``data_loader`` and update ``model`` in-place.

    The model is expected to map ``(B, T, N, 3) → (B, T, num_classes)``
    so that ``CrossEntropyLoss`` can take ``output.permute(0, 2, 1)``
    against integer labels of shape ``(B, T)``.
    """
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("clips/s", SmoothedValue(window_size=10, fmt="{value:.3f}"))

    header = f"Epoch: [{epoch}]"
    for clip, target in metric_logger.log_every(data_loader, print_freq, header):
        start = time.time()
        clip = clip.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(clip)                                # (B, T, K)
        loss = criterion(output.permute(0, 2, 1), target)   # CE expects (B, K, T)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = output.argmax(dim=-1)
            acc = (pred == target).float().mean().item()

        bs = clip.size(0)
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc"].update(acc, n=bs)
        metric_logger.meters["clips/s"].update(bs / max(time.time() - start, 1e-9))

        if lr_scheduler is not None:
            lr_scheduler.step()
        sys.stdout.flush()
