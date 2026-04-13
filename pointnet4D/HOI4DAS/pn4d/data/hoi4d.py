"""HOI4D 4D action segmentation dataset.

Each clip is a sequence of ``T = 150`` frames, ``N = 2048`` points per
frame, with one action label per frame from a 19-class taxonomy.

Files (placed under ``root``):

    train1.h5, train2.h5, train3.h5, train4.h5
    test1.h5, test2.h5
    test_wolabel.h5     # only used for challenge submission inference

Each ``.h5`` file holds three datasets::

    pcd     : (N, 150, 2048, 3) float32  — xyz coordinates
    center  : (N, 150, 3)       float32  — frame-wise centroid
    label   : (N, 150)          int64    — per-frame action class

The ``test_wolabel.h5`` file omits ``label``.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


_TRAIN_FILES: Tuple[str, ...] = ("train1.h5", "train2.h5", "train3.h5", "train4.h5")
_TEST_FILES: Tuple[str, ...] = ("test1.h5", "test2.h5")
_TEST_WOLABEL: Tuple[str, ...] = ("test_wolabel.h5",)


class HOI4DActionSeg(Dataset):
    """HOI4D action segmentation dataset.

    Args:
        root: directory containing the ``*.h5`` files described above.
        split: ``"train"``, ``"test"`` or ``"test_wolabel"``.
        files: optional list of file names overriding the default split.
        augment: whether to apply random flip / scaling / jitter (only used
            when ``split == "train"``). Set to ``False`` for ablations.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        files: Optional[List[str]] = None,
        augment: bool = True,
    ) -> None:
        super().__init__()
        if split not in ("train", "test", "test_wolabel"):
            raise ValueError(f"split must be 'train', 'test' or 'test_wolabel', got {split!r}")
        self.root = root
        self.split = split
        self.augment = augment and split == "train"

        if files is None:
            if split == "train":
                files = list(_TRAIN_FILES)
            elif split == "test":
                files = list(_TEST_FILES)
            else:
                files = list(_TEST_WOLABEL)

        pcd_chunks: List[np.ndarray] = []
        center_chunks: List[np.ndarray] = []
        label_chunks: List[np.ndarray] = []
        for fn in files:
            path = os.path.join(root, fn)
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            with h5py.File(path, "r") as f:
                pcd_chunks.append(np.asarray(f["pcd"]))
                center_chunks.append(np.asarray(f["center"]))
                if "label" in f:
                    label_chunks.append(np.asarray(f["label"]))
                else:
                    # placeholder so __getitem__ has a stable shape
                    label_chunks.append(np.zeros(pcd_chunks[-1].shape[:2], dtype=np.int64))

        self.pcd = np.concatenate(pcd_chunks, axis=0)
        self.center = np.concatenate(center_chunks, axis=0)
        self.label = np.concatenate(label_chunks, axis=0)
        assert self.pcd.shape[0] == self.label.shape[0] == self.center.shape[0]

    def __len__(self) -> int:
        return int(self.pcd.shape[0])

    def _augment(self, pc: np.ndarray, center: np.ndarray) -> np.ndarray:
        """Random flip across X, jitter, and uniform scaling around the centroid."""
        if np.random.rand() > 0.5:
            pc = pc - center
            pc[:, 0] *= -1
            pc = pc + center
        else:
            pc = pc - center
            jitter = np.clip(0.01 * np.random.randn(*pc.shape), -0.05, 0.05)
            pc = pc + jitter + center

        scale = np.random.uniform(0.8, 1.2)
        pc = (pc - center) * scale + center
        return pc

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pc = self.pcd[index]
        center0 = self.center[index][0]
        label = self.label[index]
        if self.augment:
            pc = self._augment(pc, center0)
        return (
            torch.from_numpy(pc.astype(np.float32, copy=False)),
            torch.from_numpy(label.astype(np.int64, copy=False)),
        )
