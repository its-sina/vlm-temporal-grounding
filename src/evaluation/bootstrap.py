"""Paired bootstrap confidence intervals for differences between two methods on the same queries."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.evaluation.metrics import RECALL_THRESHOLDS


def paired_bootstrap(iou_a: Sequence[float], iou_b: Sequence[float], n_boot: int = 10000, seed: int = 0,
                     clusters: Sequence[str] | None = None) -> dict:
    """Mean difference (a - b, in points) and 95% CI for mIoU and R@{0.3,0.5,0.7}.

    clusters: optional cluster id per query (e.g. the video id). Queries of one video share frames and
    are not independent; with clusters, whole videos are resampled instead of single queries.
    """
    ia, ib = np.asarray(iou_a, dtype=float), np.asarray(iou_b, dtype=float)
    if ia.shape != ib.shape:
        raise ValueError("paired inputs must have the same length")
    n = len(ia)
    rng = np.random.default_rng(seed)
    if clusters is None:
        groups = None
        idx = rng.integers(0, n, (n_boot, n))
    else:
        ids, inverse = np.unique(np.asarray(clusters), return_inverse=True)
        groups = [np.flatnonzero(inverse == g) for g in range(len(ids))]
        idx = rng.integers(0, len(groups), (n_boot, len(groups)))
    out: dict = {"n": n}
    metrics = [("mIoU", lambda x: x)] + [(f"R@{t}", lambda x, t=t: x >= t) for t in RECALL_THRESHOLDS]
    for name, f in metrics:
        d = 100 * (f(ia).astype(float) - f(ib).astype(float))
        if groups is None:
            boots = d[idx].mean(1)
        else:
            sums = np.array([d[g].sum() for g in groups])
            sizes = np.array([len(g) for g in groups])
            boots = sums[idx].sum(1) / sizes[idx].sum(1)
        out[name] = {"diff": float(d.mean()),
                     "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    return out
