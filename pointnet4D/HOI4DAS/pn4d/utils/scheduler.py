"""Learning-rate schedulers used by pn4d training scripts."""
from __future__ import annotations

from bisect import bisect_right
from typing import List

import torch


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    """MultiStepLR with linear/constant warmup at the start of training.

    The number of warmup *iterations* (not epochs) is provided so that
    schedulers can be stepped per-iteration alongside the optimizer.

    Args:
        optimizer: wrapped optimizer.
        milestones: list of iteration indices at which to decay the lr.
        gamma: multiplicative decay factor.
        warmup_factor: starting fraction of base lr at iter 0.
        warmup_iters: number of warmup iterations.
        warmup_method: ``"linear"`` or ``"constant"``.
        last_epoch: index of the last iteration; ``-1`` means start.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_factor: float = 1e-3,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ) -> None:
        if list(milestones) != sorted(milestones):
            raise ValueError(f"Milestones should be sorted, got {milestones}")
        if warmup_method not in ("constant", "linear"):
            raise ValueError(f"warmup_method must be 'constant' or 'linear', got {warmup_method!r}")

        self.milestones = list(milestones)
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self):  # type: ignore[override]
        warmup_factor = 1.0
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            else:  # linear
                alpha = float(self.last_epoch) / float(self.warmup_iters)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
