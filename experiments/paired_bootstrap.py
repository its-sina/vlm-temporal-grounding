"""Paired bootstrap CIs for metric differences between runs on their common queries.

  python experiments/paired_bootstrap.py RUN_A RUN_B [RUN_C RUN_D ...]   # consecutive pairs A-B, C-D, ...
  python experiments/paired_bootstrap.py --ref RUN_REF RUN_1 RUN_2 ...    # each run vs the reference
  add --cluster-by-video to resample whole videos (queries of one video are not independent)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.bootstrap import paired_bootstrap  # noqa: E402
from src.utils import paths  # noqa: E402
from src.utils.io import load_predictions  # noqa: E402


def compare(root: Path, a: str, b: str, n_boot: int, cluster: bool) -> dict:
    pa, pb = load_predictions(root / a), load_predictions(root / b)
    uids = sorted(set(pa) & set(pb))
    if not uids:
        raise SystemExit(f"{a} and {b} have no queries in common")
    clusters = [pa[u]["vid"] for u in uids] if cluster else None
    return paired_bootstrap([pa[u]["iou"] for u in uids], [pb[u]["iou"] for u in uids], n_boot=n_boot,
                            clusters=clusters)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref", help="compare every run against this one")
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--cluster-by-video", action="store_true")
    a = ap.parse_args(argv)
    root = a.results_dir or paths.results_dir()
    if a.ref:
        pairs = [(r, a.ref) for r in a.runs]
    else:
        if len(a.runs) % 2:
            ap.error("give an even number of runs (pairs) or use --ref")
        pairs = list(zip(a.runs[::2], a.runs[1::2]))
    for x, y in pairs:
        o = compare(root, x, y, a.n_boot, a.cluster_by_video)
        print(f"{x} - {y} (n={o['n']}): " + "  ".join(
            f"{k} {o[k]['diff']:+.2f} [{o[k]['ci95'][0]:+.2f},{o[k]['ci95'][1]:+.2f}]"
            for k in ["mIoU", "R@0.3", "R@0.5", "R@0.7"]))


if __name__ == "__main__":
    main()
