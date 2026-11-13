import numpy as np
import pytest

from src.evaluation.bootstrap import paired_bootstrap
from src.evaluation.metrics import compute_iou, summarize
from src.grounding.parsing import frames_to_seconds, parse_timestamps, parse_timestamps_float, unit_aware_span


@pytest.mark.parametrize("text,int_span,float_span", [
    ("From 3 to 9.", [3, 9], [3.0, 9.0]),
    ("from frame 12 to frame 20", [12, 20], [12.0, 20.0]),
    ("from Frame 18 to Frame 25", [18, 25], [18.0, 25.0]),
    ("Frames 4 and 11", [4, 11], [4.0, 11.0]),
    ("between frames 6 - 14", [6, 14], [6.0, 14.0]),
    ("20.6 - 31.0 seconds", [20, 31], [20.6, 31.0]),
    ("12s-18s", [12, 18], [12.0, 18.0]),
    ("from 9 to 3", [3, 9], [3.0, 9.0]),                     # reversed endpoints are ordered
    ("The moment is at 5.", [0, 0], [0.0, 0.0]),              # a single number is not a span
    ("I cannot determine this.", [0, 0], [0.0, 0.0]),
    ("", [0, 0], [0.0, 0.0]),
    ("Frames 2-4 overlap; final answer: from 10 to 12", [10, 12], [10.0, 12.0]),  # 'from x to y' first
    ("from 3.9 to 7.2", [3, 7], [3.9, 7.2]),                  # integer parser truncates
    ("FROM 1 TO 2", [1, 2], [1.0, 2.0]),
])
def test_parsers(text, int_span, float_span):
    assert parse_timestamps(text) == int_span
    assert parse_timestamps_float(text) == pytest.approx(float_span)


def test_iou_edge_cases():
    assert compute_iou([0, 10], [5, 15]) == pytest.approx(5 / 15)
    assert compute_iou([0, 0], [3, 5]) == 0.0
    assert compute_iou([3, 5], [3, 5]) == 1.0
    assert compute_iou([4, 4], [4, 4]) == 0.0           # empty union
    assert compute_iou([0, 10], [5, 6]) == pytest.approx(0.1)


def test_summarize():
    m = summarize([0.8, 0.4, 0.0])
    assert m["n"] == 3
    assert m["R@0.3"] == pytest.approx(200 / 3) and m["R@0.5"] == pytest.approx(100 / 3)
    assert m["R@0.7"] == pytest.approx(100 / 3) and m["mIoU"] == pytest.approx(40.0)
    assert summarize([]) == {"n": 0}


def test_unit_conversion():
    assert frames_to_seconds([4, 10], 2.0) == [2.0, 5.0]
    assert unit_aware_span([10, 20], 60.0, 0.5) == [20.0, 40.0]   # plausible frame indices
    assert unit_aware_span([40, 58], 60.0, 0.5) == [40, 58]       # 58 > 31 + 1: seconds


def _reference_bootstrap(ia, ib, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ia), (n_boot, len(ia)))
    d = 100 * (ia - ib)
    bs = d[idx].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def test_paired_bootstrap_matches_reference_and_clusters():
    rng = np.random.default_rng(1)
    ia, ib = rng.random(200), rng.random(200)
    out = paired_bootstrap(ia, ib, n_boot=2000)
    diff, lo, hi = _reference_bootstrap(ia, ib)
    assert out["mIoU"]["diff"] == pytest.approx(diff) and out["mIoU"]["ci95"] == pytest.approx([lo, hi])
    same = paired_bootstrap(ia, ia, n_boot=200, clusters=[str(i // 4) for i in range(200)])
    assert same["mIoU"] == {"diff": 0.0, "ci95": [0.0, 0.0]}
    clustered = paired_bootstrap(ia, ib, n_boot=2000, clusters=[str(i // 4) for i in range(200)])
    assert clustered["mIoU"]["diff"] == pytest.approx(diff)
