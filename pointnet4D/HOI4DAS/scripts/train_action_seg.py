"""Unified DDP training script for pn4d action segmentation models.

Launch on a single node with N GPUs::

    torchrun --nproc_per_node=N scripts/train_action_seg.py \\
        --config configs/pointnet4d_hoi4d.yaml \\
        --output-dir runs/pointnet4d

Single-process (no DDP) also works::

    python scripts/train_action_seg.py --config configs/pointnet4d_hoi4d.yaml

Any YAML field can be overridden from the command line, e.g.
``--epochs 5 --batch-size 4``.
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
from pn4d.engine import evaluate, train_one_epoch
from pn4d.models import P4Transformer, PointNet4D, PointNet4DPlus
from pn4d.utils import (
    WarmupMultiStepLR,
    init_distributed_mode,
    is_main_process,
    load_checkpoint,
    save_checkpoint,
)


_MODEL_REGISTRY = {
    "p4_transformer": P4Transformer,
    "pointnet4d": PointNet4D,
    "pointnet4d_plus": PointNet4DPlus,
}


def build_model(name: str, kwargs: Dict[str, Any]) -> nn.Module:
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}; choose from {list(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[name](**kwargs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="pn4d action segmentation training")
    p.add_argument("--config", required=True, type=str, help="YAML config path")
    p.add_argument("--output-dir", default=None, type=str, help="checkpoint / log dir")
    p.add_argument("--data-path", default=None, type=str, help="override data root")
    p.add_argument("--epochs", default=None, type=int)
    p.add_argument("--batch-size", default=None, type=int)
    p.add_argument("--workers", default=None, type=int)
    p.add_argument("--lr", default=None, type=float)
    p.add_argument("--resume", default=None, type=str, help="checkpoint to resume from")
    p.add_argument("--pretrained", default=None, type=str,
                   help="path to a 4DMAP pretrain checkpoint to load into the encoder")
    p.add_argument("--seed", default=None, type=int)
    return p.parse_args()


def merge_config(args: argparse.Namespace) -> Dict[str, Any]:
    with open(args.config, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    cli_overrides = {
        "output_dir": args.output_dir,
        "data_path": args.data_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "lr": args.lr,
        "resume": args.resume,
        "pretrained": args.pretrained,
        "seed": args.seed,
    }
    for k, v in cli_overrides.items():
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
        print(f"=== pn4d {pn4d.__version__} ===")
        print(f"config: {args.config}")
        for k, v in cfg.items():
            print(f"  {k}: {v}")

    # ---- data ----
    train_set = HOI4DActionSeg(root=cfg["data_path"], split="train", augment=True)
    test_set = HOI4DActionSeg(root=cfg["data_path"], split="test", augment=False)

    if getattr(args, "distributed", False):
        train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True)
        test_sampler = DistributedSampler(test_set, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg["workers"],
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        sampler=test_sampler,
        num_workers=cfg["workers"],
        pin_memory=True,
    )

    # ---- model ----
    model = build_model(cfg["model"]["name"], cfg["model"]["args"])
    model.to(device)

    if cfg.get("pretrained"):
        if is_main_process():
            print(f"loading pretrained encoder from {cfg['pretrained']}")
        load_checkpoint(model, cfg["pretrained"], encoder_prefix="encoder", strict=False)

    if getattr(args, "distributed", False):
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )

    criterion = nn.CrossEntropyLoss()

    # ---- optim ----
    optim_name = cfg.get("optimizer", "sgd").lower()
    if optim_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg.get("weight_decay", 0.05),
            betas=tuple(cfg.get("betas", [0.9, 0.95])),
        )
    else:  # sgd (default)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg["lr"],
            momentum=cfg.get("momentum", 0.9),
            weight_decay=cfg.get("weight_decay", 1e-4),
        )

    iters_per_epoch = len(train_loader)
    warmup_iters = cfg.get("lr_warmup_epochs", 5) * iters_per_epoch
    lr_milestones = [iters_per_epoch * m for m in cfg.get("lr_milestones", [20, 35])]
    lr_scheduler = WarmupMultiStepLR(
        optimizer,
        milestones=lr_milestones,
        gamma=cfg.get("lr_gamma", 0.5),
        warmup_factor=1e-5,
        warmup_iters=warmup_iters,
    )

    # ---- resume ----
    start_epoch = 0
    if cfg.get("resume"):
        ckpt = load_checkpoint(model, cfg["resume"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "lr_scheduler" in ckpt:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1

    # ---- train loop ----
    best_acc = 0.0
    start_time = time.time()
    for epoch in range(start_epoch, cfg["epochs"]):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_one_epoch(
            model, criterion, optimizer, lr_scheduler, train_loader,
            device, epoch, cfg.get("print_freq", 20),
        )

        # evaluate every epoch on rank 0 (cheap enough; saves restoring sampler state)
        if is_main_process() or getattr(args, "distributed", False):
            metrics = evaluate(
                model, criterion, test_loader, device,
                overlaps=tuple(cfg.get("eval_overlaps", [0.1, 0.25, 0.5])),
                print_freq=cfg.get("print_freq", 20),
            )
            cur_acc = metrics["acc"]
            best_acc = max(best_acc, cur_acc)
            if is_main_process():
                print(f"[epoch {epoch}] best_acc={best_acc:.4f}")

        # checkpoint
        if is_main_process() and cfg.get("output_dir") and (
            (epoch + 1) % cfg.get("ckpt_freq", 5) == 0 or epoch == cfg["epochs"] - 1
        ):
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "config": cfg,
            }
            save_checkpoint(state, cfg["output_dir"], f"model_{epoch}.pth")

    if is_main_process():
        elapsed = time.time() - start_time
        print(f"Total training time: {elapsed/3600:.2f} h")
        print(f"Best accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
