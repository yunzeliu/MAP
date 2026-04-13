"""Action-segmentation metrics: per-frame accuracy, edit score, F1@k.

These reproduce the standard MS-TCN protocol used by P4Transformer / PPTr
on HOI4D. Inputs to ``edit_score`` and ``f_score`` are 1-D arrays of frame-
wise predictions and ground-truth labels.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def get_labels_start_end_time(
    frame_wise_labels: Sequence[int],
    bg_class: Sequence = ("background",),
) -> Tuple[List[int], List[int], List[int]]:
    """Compress a per-frame label sequence into ``(labels, starts, ends)``."""
    labels: List[int] = []
    starts: List[int] = []
    ends: List[int] = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(len(frame_wise_labels) - 1)
    return labels, starts, ends


def _levenshtein(p: Sequence[int], y: Sequence[int], norm: bool = False) -> float:
    m, n = len(p), len(y)
    D = np.zeros((m + 1, n + 1), dtype=np.float64)
    D[:, 0] = np.arange(m + 1)
    D[0, :] = np.arange(n + 1)
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + 1)
    if norm:
        return (1 - D[-1, -1] / max(m, n)) * 100
    return D[-1, -1]


def edit_score(
    recognized: Sequence[int],
    ground_truth: Sequence[int],
    norm: bool = True,
    bg_class: Sequence = ("background",),
) -> float:
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return float(_levenshtein(P, Y, norm))


def f_score(
    recognized: Sequence[int],
    ground_truth: Sequence[int],
    overlap: float,
    bg_class: Sequence = ("background",),
) -> Tuple[float, float, float]:
    """Return ``(tp, fp, fn)`` for a single sample at the given IoU threshold."""
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0
    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * np.array(
            [p_label[j] == y_label[x] for x in range(len(y_label))]
        )
        idx = int(np.array(IoU).argmax())
        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)
