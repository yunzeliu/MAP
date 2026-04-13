"""Lightweight wrappers around ``torch.distributed`` for DDP launching.

Usage:

    # In your training script
    from pn4d.utils import init_distributed_mode, is_main_process

    init_distributed_mode(args)            # reads env vars set by torchrun
    if is_main_process():
        ...

Launch with:

    torchrun --nproc_per_node=4 scripts/train_action_seg.py ...
"""
from __future__ import annotations

import os
from argparse import Namespace
from typing import Optional

import torch
import torch.distributed as dist


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def setup_for_distributed(is_master: bool) -> None:
    """Suppress ``print`` on non-master ranks (still allow ``force=True``)."""
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):  # type: ignore[no-redef]
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args: Optional[Namespace] = None) -> None:
    """Initialize ``torch.distributed`` from environment variables.

    Compatible with ``torchrun`` (``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``)
    and SLURM (``SLURM_PROCID``). When neither is set, the call is a no-op
    and the script runs as a single process.
    """
    if args is None:
        args = Namespace()

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ.get("SLURM_NTASKS", 1))
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}/{args.world_size}, gpu {args.gpu})", flush=True)

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=os.environ.get("DIST_URL", "env://"),
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)
