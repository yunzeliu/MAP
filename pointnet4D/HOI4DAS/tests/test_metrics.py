"""Unit tests for the action-segmentation metrics."""
from __future__ import annotations

import numpy as np

from pn4d.engine import segmentation_metrics
from pn4d.utils.metrics import edit_score, f_score, get_labels_start_end_time


def test_get_labels_start_end_time_single_segment():
    labels = [0, 0, 1, 1, 1, 2, 2]
    L, S, E = get_labels_start_end_time(labels, bg_class=())
    assert L == [0, 1, 2]
    assert S == [0, 2, 5]
    assert E == [2, 5, 6]


def test_edit_score_perfect():
    pred = [1, 1, 2, 2, 3, 3]
    gt = [1, 1, 2, 2, 3, 3]
    assert edit_score(pred, gt, bg_class=()) == 100.0


def test_edit_score_substitution():
    pred = [1, 1, 9, 9, 3, 3]   # one substitution: 2 -> 9
    gt = [1, 1, 2, 2, 3, 3]
    score = edit_score(pred, gt, bg_class=())
    assert 0 < score < 100


def test_f_score_perfect_match():
    pred = [1, 1, 2, 2]
    gt = [1, 1, 2, 2]
    tp, fp, fn = f_score(pred, gt, overlap=0.5, bg_class=())
    assert (tp, fp, fn) == (2.0, 0.0, 0.0)


def test_segmentation_metrics_aggregator_perfect():
    # Build sequences with clearly defined multi-frame segments so the IoU
    # numerator/denominator are well-behaved (no length-0 segments).
    targets = np.array(
        [[0] * 4 + [1] * 4 + [2] * 4 + [3] * 4] * 4,
        dtype=np.int32,
    )
    metrics = segmentation_metrics(targets, targets, overlaps=(0.1, 0.25, 0.5))
    assert metrics["acc"] == 1.0
    assert metrics["f1@0.10"] == 100.0
    assert metrics["f1@0.50"] == 100.0
