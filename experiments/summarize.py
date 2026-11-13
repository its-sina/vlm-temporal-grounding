"""Aggregate prediction files into metrics.

For every run directory containing pred_shard*.jsonl, writes <run>/metrics.json and an overview table
<results>/summary.csv. ActivityNet-style runs (fps != 1 with T2SGrid labels) additionally get
unit-aware and reference-protocol scores.

  python experiments/summarize.py [--results-dir results]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import compute_iou, summarize  # noqa: E402
from src.grounding.parsing import unit_aware_span  # noqa: E402
from src.utils import paths  # noqa: E402
from src.utils.io import load_predictions  # noqa: E402

MEAN_KEYS = ["n_input_tokens", "n_visual_tokens", "n_visual_tokens_encoded", "t_vision", "t_prefill",
             "t_decode", "t_model", "t_preprocess", "t_total"]
COLUMNS = ["exp", "dataset", "subset", "n", "method", "grid", "R@0.3", "R@0.5", "R@0.7", "mIoU",
           "n_input_tokens", "n_visual_tokens", "n_visual_tokens_encoded", "t_vision", "t_prefill", "t_decode",
           "t_model", "peak_mem_gb", "parse_fail"]


def run_metrics(run_dir: Path) -> dict | None:
    recs = list(load_predictions(run_dir).values())
    if not recs:
        return None
    cfg_path = run_dir / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    m = summarize([r["iou"] for r in recs])
    for k in MEAN_KEYS:
        v = [r[k] for r in recs if r.get(k) is not None]
        m[k] = float(np.mean(v)) if v else float("nan")
    v = [r["peak_mem_gb"] for r in recs if r.get("peak_mem_gb") is not None]
    m["peak_mem_gb"] = float(np.max(v)) if v else float("nan")
    fps = cfg.get("fps", 1.0)
    if fps != 1.0 and cfg.get("labels", "upstream") == "upstream" and all("pred_raw" in r for r in recs):
        unit_aware = summarize([compute_iou(unit_aware_span(r["pred_raw"], r["duration"], fps), r["gt"]) for r in recs])
        reference = summarize([r["iou_upstream"] for r in recs])
        m.update({f"unitaware_{k}": v for k, v in unit_aware.items() if k != "n"})
        m.update({f"upstream_{k}": v for k, v in reference.items() if k != "n"})
    m["parse_fail"] = sum(r["pred"] == [0, 0] for r in recs)
    m.update(exp=run_dir.name, method=cfg.get("method"), grid=cfg.get("grid"), dataset=cfg.get("dataset"),
             subset=cfg.get("subset"))
    (run_dir / "metrics.json").write_text(json.dumps(m, indent=1))
    return m


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=None, help="default: $VTG_RESULTS_DIR or ./results")
    a = ap.parse_args(argv)
    root = a.results_dir or paths.results_dir()
    rows = [m for d in sorted(p for p in root.iterdir() if p.is_dir()) if (m := run_metrics(d))]
    if not rows:
        print(f"no prediction files under {root}")
        return
    with open(root / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)  # noqa: E731
    show = ["exp", "n", "method", "R@0.3", "R@0.5", "R@0.7", "mIoU", "n_visual_tokens", "t_model"]
    print("  ".join(f"{c:>12}" for c in show))
    for r in rows:
        print("  ".join(f"{fmt(r.get(c)):>12}" for c in show))
    print(f"wrote {root / 'summary.csv'}")


if __name__ == "__main__":
    main()
