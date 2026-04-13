"""DDP-enabled 4DMAP pretraining for pn4d backbones.

Launch::

    torchrun --nproc_per_node=N scripts/pretrain_4dmap.py \\
        --config configs/4dmap_pretrain_hoi4d.yaml \\
        --output-dir runs/4dmap_pretrain
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import pn4d
from pn4d.data import HOI4DActionSeg
from pn4d.engine import pretrain_one_epoch
from pn4d.models import PointNet4D, PointNet4DPlus
from pn4d.pretrain import FourDMAP

_ENCODER_REGISTRY = {
    "pointnet4d": PointNet4D,
    "pointnet4d_plus": PointNet4DPlus,
}
from pn4d.utils import (
    WarmupMultiStepLR,
    init_distributed_mode,
    is_main_process,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="pn4d 4DMAP pretraining")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--output-dir", default=None, type=str)
    p.add_argument("--data-path", default=None, type=str)
    p.add_argument("--epochs", default=None, type=int)
    p.add_argument("--batch-size", default=None, type=int)
    p.add_argument("--workers", default=None, type=int)
    p.add_argument("--lr", default=None, type=float)
    p.add_argument("--mask-ratio", default=None, type=float)
    p.add_argument("--seed", default=None, type=int)
    return p.parse_args()


def merge_config(args: argparse.Namespace) -> Dict[str, Any]:
    with open(args.config, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)
    overrides = {
        "output_dir": args.output_dir,
        "data_path": args.data_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "lr": args.lr,
        "mask_ratio": args.mask_ratio,
        "seed": args.seed,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


def main() -> None:
    args = parse_args()
    cfg = merge_config(args)

    init_distributed_mode(args)
    device = torch.device("cuda")

    seed = int(cfg.get("seed", 0)) + getattr(args, "rank", 0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    if cfg.get("output_dir"):
        Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    if is_main_process():
        print(f"=== pn4d {pn4d.__version__} pretrain ===")
        for k, v in cfg.items():
            print(f"  {k}: {v}")

    # ---- data: only train split (use ALL clips, labels are ignored) ----
    train_set = HOI4DActionSeg(root=cfg["data_path"], split="train", augment=True)
    if getattr(args, "distributed", False):
        train_sampler: DistributedSampler | None = DistributedSampler(
            train_set, shuffle=True, drop_last=True
        )
    else:
        train_sampler = None
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg["workers"],
        pin_memory=True,
        drop_last=True,
    )

    # ---- model: encoder + 4DMAP wrapper ----
    encoder_args = cfg["encoder"]["args"]
    encoder_name = cfg["encoder"].get("name", "pointnet4d")
    EncoderCls = _ENCODER_REGISTRY.get(encoder_name, PointNet4D)
    encoder = EncoderCls(**encoder_args)
    feat_dim = encoder_args.get("feat_dim", encoder_args.get("dim", 1024))
    model = FourDMAP(
        encoder=encoder,
        feat_dim=feat_dim,
        num_points=cfg["num_points"],
        decoder_depth=cfg.get("decoder_depth", 4),
        decoder_heads=cfg.get("decoder_heads", 8),
        mask_ratio=cfg.get("mask_ratio", 0.5),
    ).to(device)

    if getattr(args, "distributed", False):
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.05),
        betas=(0.9, 0.95),
    )
    iters_per_epoch = len(train_loader)
    warmup_iters = cfg.get("lr_warmup_epochs", 5) * iters_per_epoch
    lr_scheduler = WarmupMultiStepLR(
        optimizer,
        milestones=[iters_per_epoch * m for m in cfg.get("lr_milestones", [50, 80])],
        gamma=cfg.get("lr_gamma", 0.1),
        warmup_factor=1e-5,
        warmup_iters=warmup_iters,
    )

    start_time = time.time()
    for epoch in range(cfg["epochs"]):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        pretrain_one_epoch(
            model, optimizer, lr_scheduler, train_loader,
            device, epoch, cfg.get("print_freq", 20),
        )
        if is_main_process() and cfg.get("output_dir") and (
            (epoch + 1) % cfg.get("ckpt_freq", 10) == 0 or epoch == cfg["epochs"] - 1
        ):
            # save the encoder by itself so it can be loaded for finetune
            unwrapped = model.module if hasattr(model, "module") else model
            save_checkpoint(
                {
                    "model": {f"encoder.{k}": v for k, v in unwrapped.encoder.state_dict().items()},
                    "epoch": epoch,
                    "config": cfg,
                },
                cfg["output_dir"],
                f"4dmap_encoder_{epoch}.pth",
            )

    if is_main_process():
        elapsed = time.time() - start_time
        print(f"Total pretraining time: {elapsed/3600:.2f} h")


if __name__ == "__main__":
    main()
