"""Temporal grounding metrics: IoU, Recall@IoU and mean IoU (all in percent)."""

from __future__ import annotations

from typing import Sequence

RECALL_THRESHOLDS = (0.3, 0.5, 0.7)


def compute_iou(span: Sequence[float], gt: Sequence[float]) -> float:
    """Temporal IoU of two [start, end] intervals; 0 for an empty union."""
    a, b = span
    c, d = gt
    inter = max(0.0, min(b, d) - max(a, c))
    union = max(b, d) - min(a, c)
    return inter / union if union > 0 else 0.0


def summarize(ious: Sequence[float]) -> dict:
    """``{"n", "R@0.3", "R@0.5", "R@0.7", "mIoU"}`` in percent."""
    n = len(ious)
    if n == 0:
        return {"n": 0}
    out = {"n": n}
    for t in RECALL_THRESHOLDS:
        out[f"R@{t}"] = 100 * sum(i >= t for i in ious) / n
    out["mIoU"] = 100 * sum(ious) / n
    return out
