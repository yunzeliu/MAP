"""Checkpoint helpers: rank-aware save and partial-state load.

The loader supports stripping ``module.`` prefixes (DDP) and an
``encoder_prefix`` so that pretrained backbones can be reused for finetune.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from pn4d.utils.distributed import is_main_process


def save_checkpoint(
    state: Dict[str, Any],
    output_dir: str,
    name: str,
) -> Optional[str]:
    """Save ``state`` only from the main process. Returns the file path or ``None``."""
    if not is_main_process():
        return None
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    torch.save(state, path)
    return path


def load_checkpoint(
    model: nn.Module,
    ckpt_path: str,
    *,
    encoder_prefix: Optional[str] = None,
    strict: bool = False,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Load a checkpoint into ``model`` with prefix-aware mapping.

    Args:
        model: target module (may be wrapped with ``DDP`` / ``DataParallel``).
        ckpt_path: path to a ``.pth`` checkpoint produced by pn4d.
        encoder_prefix: if set, only state-dict entries starting with this
            prefix are loaded (after stripping the prefix). Useful for
            loading a pretrained encoder into a finetune model whose encoder
            sub-module has a different parent name.
        strict: forwarded to :py:meth:`nn.Module.load_state_dict`.
        map_location: where to map tensors when loading.
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    # strip ``module.`` (DDP/DataParallel)
    state_dict = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state_dict.items()}

    if encoder_prefix is not None:
        prefix = encoder_prefix.rstrip(".") + "."
        state_dict = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}

    # also strip ``module.`` from target if needed
    target = model.module if hasattr(model, "module") else model
    incompatible = target.load_state_dict(state_dict, strict=strict)

    if not strict:
        if incompatible.missing_keys:
            print(f"[load_checkpoint] missing keys: {len(incompatible.missing_keys)}")
        if incompatible.unexpected_keys:
            print(f"[load_checkpoint] unexpected keys: {len(incompatible.unexpected_keys)}")

    return ckpt if isinstance(ckpt, dict) else {}
