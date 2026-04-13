"""Single-epoch 4DMAP pretraining loop."""
from __future__ import annotations

import sys
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pn4d.utils.logger import MetricLogger, SmoothedValue


def pretrain_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    print_freq: int = 20,
) -> None:
    """Self-supervised epoch — model is a :class:`pn4d.pretrain.FourDMAP`
    that returns ``(loss, preds)`` from its ``forward``.
    """
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("clips/s", SmoothedValue(window_size=10, fmt="{value:.3f}"))

    header = f"Pretrain Epoch: [{epoch}]"
    for clip, _ in metric_logger.log_every(data_loader, print_freq, header):
        start = time.time()
        clip = clip.to(device, non_blocking=True)
        loss, _ = model(clip)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = clip.size(0)
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["clips/s"].update(bs / max(time.time() - start, 1e-9))

        if lr_scheduler is not None:
            lr_scheduler.step()
        sys.stdout.flush()
