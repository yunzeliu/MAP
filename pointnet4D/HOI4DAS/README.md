# pn4d

A modular toolkit for 4D point cloud video understanding, featuring the **PointNet4D** backbone and **4DMAP** self-supervised pretraining framework.

## Models

| Model | Spatial Backbone | Temporal Fusion |
|-------|-----------------|-----------------|
| `P4Transformer` | Point4DConv | Transformer |
| `PointNet4D` | PointNet++ | Mamba + Transformer |
| `PointNet4D++` | Point4DConv | Mamba + Transformer |

**4DMAP** is a frame-masked autoregressive pretraining method that can be applied to any of the above backbones.

## Installation

### Requirements

- Python >= 3.9
- PyTorch >= 2.0 with CUDA
- A working `nvcc` compiler

### Install

```bash
# 1. Install PyTorch (match your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. Install Mamba (required for PointNet4D / PointNet4D++)
pip install causal-conv1d mamba-ssm --no-build-isolation

# 3. Install pn4d (builds the bundled CUDA extension)
git clone https://github.com/yunzeliu/pn4d.git && cd pn4d
TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0;12.0" pip install -e . --no-build-isolation
```

Set `TORCH_CUDA_ARCH_LIST` to match your GPU architecture (e.g., `8.0` for A100, `9.0` for H100, `12.0` for Blackwell).

## Quick Start

### Python API

```python
import pn4d
import torch

# Build a model
model = pn4d.models.PointNet4D(num_classes=19, feat_dim=1024, depth=5).cuda()

# Forward pass: (batch, frames, points, 3)
x = torch.randn(2, 150, 2048, 3, device="cuda")
logits = model(x)  # (2, 150, 19)
```

Available models:
```python
pn4d.models.P4Transformer(num_classes=19, dim=2048, depth=5, ...)
pn4d.models.PointNet4D(num_classes=19, feat_dim=1024, depth=5, ...)
pn4d.models.PointNet4DPlus(num_classes=19, dim=1024, depth=5, ...)
```

### Dataset

The HOI4D action segmentation dataset consists of HDF5 files. Place them under a directory:

```
<data_root>/
  train1.h5  train2.h5  train3.h5  train4.h5
  test1.h5   test2.h5
```

```python
from pn4d.data import HOI4DActionSeg

train_set = HOI4DActionSeg(root="<data_root>", split="train")
test_set  = HOI4DActionSeg(root="<data_root>", split="test")
```

## Training

All training is driven by YAML configs in `configs/`.

### Single GPU

```bash
python scripts/train_action_seg.py \
    --config configs/pointnet4d_plus_hoi4d.yaml \
    --output-dir runs/pointnet4d_plus
```

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=4 scripts/train_action_seg.py \
    --config configs/pointnet4d_plus_hoi4d.yaml \
    --output-dir runs/pointnet4d_plus
```

### CLI Overrides

Any config field can be overridden from the command line:

```bash
python scripts/train_action_seg.py \
    --config configs/pointnet4d_hoi4d.yaml \
    --output-dir runs/my_run \
    --epochs 100 --batch-size 4 --lr 0.001
```

## 4DMAP Pretraining

4DMAP is a frame-level masked autoregressive pretraining strategy. It randomly masks frames, encodes the sequence with the backbone, and uses a causal Transformer decoder to reconstruct the masked frames.

### Step 1: Pretrain

```bash
python scripts/pretrain_4dmap.py \
    --config configs/4dmap_pretrain_hoi4d.yaml \
    --output-dir runs/4dmap_pretrain
```

This saves encoder checkpoints as `4dmap_encoder_<epoch>.pth`.

### Step 2: Finetune

```bash
python scripts/train_action_seg.py \
    --config configs/pointnet4d_plus_hoi4d.yaml \
    --output-dir runs/4dmap_finetune \
    --pretrained runs/4dmap_pretrain/4dmap_encoder_49.pth
```

The `--pretrained` flag loads only the encoder weights (keys prefixed with `encoder.`) into the model, leaving the classification head randomly initialized.

## Available Configs

| Config | Model |
|--------|-------|
| `p4_transformer_hoi4d.yaml` | P4Transformer |
| `pointnet4d_hoi4d.yaml` | PointNet4D |
| `pointnet4d_plus_hoi4d.yaml` | PointNet4D++ |
| `4dmap_pretrain_hoi4d.yaml` | 4DMAP pretraining |

## Project Structure

```
pn4d/
  __init__.py
  ops/                    # CUDA kernels: FPS, ball query, grouping, interpolation
    _ext_src/             # C++/CUDA source files
    pointnet2_utils.py    # Autograd function wrappers
  modules/                # Reusable building blocks
    point_4d_conv.py      # Point4DConv (spatiotemporal tube convolution)
    pointnet_pp_extractor.py  # Per-frame PointNet++ encoder
    temporal_layers.py    # Mamba3DBlock, causal Attention, PointNet4DTemporalFusion
    transformer.py        # Standard pre-LN Transformer block
  models/                 # Full models (input -> output)
    p4_transformer.py
    pointnet4d.py
    pointnet4d_plus.py
  pretrain/               # Self-supervised pretraining
    decoder.py            # Causal AR Transformer decoder
    fourd_map.py          # 4DMAP wrapper (encoder + decoder + CD-L2 loss)
  data/
    hoi4d.py              # HOI4D action segmentation dataset
  engine/                 # Training and evaluation loops
    trainer.py
    evaluator.py
    pretrainer.py
  utils/
    distributed.py        # DDP setup helpers
    logger.py             # MetricLogger
    metrics.py            # Edit score, F1 score
    scheduler.py          # WarmupMultiStepLR
    checkpoint.py         # Save/load with prefix-aware mapping
scripts/
  train_action_seg.py     # Unified training script (single/multi-GPU)
  pretrain_4dmap.py       # 4DMAP pretraining script
configs/                  # YAML training configurations
tests/                    # Unit tests
```

## Tests

```bash
pip install pytest
pytest tests/ -q
```

## Citation

```bibtex
@inproceedings{liu2026pointnet4d,
  title={PointNet4D: A Lightweight 4D Point Cloud Video Backbone for Online and Offline Perception in Robotic Applications},
  author={Liu, Yunze and Wang, Zifan and Wu, Peiran and Ao, Jiayang},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={3203--3212},
  year={2026}
}

@inproceedings{liu2025map,
  title={MAP: Unleashing Hybrid Mamba-Transformer Vision Backbone's Potential with Masked Autoregressive Pretraining},
  author={Liu, Yunze and Yi, Li},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9676--9685},
  year={2025}
}
```

## Acknowledgement

This codebase was refactored and rewritten with the assistance of [Claude Code](https://claude.ai/code).

## License

MIT
